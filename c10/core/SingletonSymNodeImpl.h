#include <c10/core/ConstantSymNodeImpl.h>
#include <c10/core/SymBool.h>
#include <c10/core/SymNodeImpl.h>
#include <iostream>

namespace c10 {

// An int-like object that only defines the equality operator.
class C10_API SingletonSymNodeImpl : public SymNodeImpl {
 public:
  // CAUTION: you should probably not be constructing these directly; please
  // the higher-level API in python instead (TODO: actually introduce that).
  explicit SingletonSymNodeImpl(int64_t val) : val_(val) {}

  bool bool_() override {
    return false;
  }

  bool is_int() override {
    return true;
  }

  bool is_float() override {
    return false;
  }

  bool is_bool() override {
    return false;
  }

  bool has_hint() override {
    return true;
  }

  c10::SymNode wrap_int(int64_t num) override {
    return SymNode(c10::make_intrusive<ConstantSymNodeImpl<int64_t>>(num));
  };

  int64_t guard_int(const char* file, int64_t line) override {
    // TODO: when is this used?
    TORCH_CHECK(false);
  }

  double guard_float(const char* file, int64_t line) override {
    TORCH_CHECK(false, "not a float");
  }

  bool guard_bool(const char* file, int64_t line) override {
    TORCH_CHECK(false, "not a bool");
  }

  int64_t int_() override {
    // TODO: when is this used?
    TORCH_CHECK(false);
  }

  std::string str() override {
    return "j" + std::to_string(val_);
  }

  c10::SymNode eq(const c10::SymNode& other) override {
    c10::optional<int64_t> c = other->singleton_int();
    bool ret = c.has_value() && val_ == *c;
    return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(ret));
  }

  c10::SymNode ne(const c10::SymNode& other) override {
    c10::optional<int64_t> c = other->singleton_int();
    bool ret = c.has_value() && val_ != *c;
    return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(ret));
  }

  // Would be cool to have the ability to arbitrarily constrain the range of the
  // singleton's values. For now we constrain the range to be [2, int64_t::max()]
  // (inclusive), as to bypass 0/1 specialization checks.
  c10::SymNode ge(const c10::SymNode& other) override {
    if (other->singleton_int().has_value()) {
        return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(false));
    }
    c10::optional<int64_t> c = other->constant_int();
    TORCH_CHECK(c.has_value());
    return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(*c <= 2));
  }

  c10::SymNode lt(const c10::SymNode& other) override {
    return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(false));
  }

  c10::optional<int64_t> singleton_int() override {
    return val_;
  }

#define DEFINE_BINARY_NOT_SUPPORTED(name)                           \
  c10::SymNode name(const c10::SymNode& other) override {           \
    TORCH_CHECK(false, #name " not supported by SingletonSymNode"); \
  }

  DEFINE_BINARY_NOT_SUPPORTED(add)
  DEFINE_BINARY_NOT_SUPPORTED(sub)
  DEFINE_BINARY_NOT_SUPPORTED(mul)
  DEFINE_BINARY_NOT_SUPPORTED(truediv)
  DEFINE_BINARY_NOT_SUPPORTED(pow)
  DEFINE_BINARY_NOT_SUPPORTED(floordiv)
  DEFINE_BINARY_NOT_SUPPORTED(mod)
  DEFINE_BINARY_NOT_SUPPORTED(gt)
  DEFINE_BINARY_NOT_SUPPORTED(sym_min)
  DEFINE_BINARY_NOT_SUPPORTED(sym_max)
  DEFINE_BINARY_NOT_SUPPORTED(sym_and)
  DEFINE_BINARY_NOT_SUPPORTED(sym_or)

#undef DEFINE_BINARY_NOT_SUPPORTED

#define DEFINE_NOT_SUPPORTED(name)                                     \
  c10::SymNode name() override {                                       \
    TORCH_CHECK(false, #name " is not supported by SingletonSymNode"); \
  }

  DEFINE_NOT_SUPPORTED(sym_not)
  DEFINE_NOT_SUPPORTED(ceil)
  DEFINE_NOT_SUPPORTED(floor)
  DEFINE_NOT_SUPPORTED(neg)
  DEFINE_NOT_SUPPORTED(clone)
  DEFINE_NOT_SUPPORTED(sym_float)

#undef DEFINE_NOT_SUPPORTED

 private:
  int64_t val_;
};

} // namespace c10
