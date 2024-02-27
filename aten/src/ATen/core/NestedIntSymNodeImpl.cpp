#include <ATen/core/NestedIntSymNodeImpl.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/util/Exception.h>

namespace c10 {

namespace {
bool _eq(const char* op, c10::SymNodeImpl* lhs, c10::SymNodeImpl* rhs) {
  TORCH_INTERNAL_ASSERT(lhs->is_nested_int());
  c10::optional<int64_t> c = rhs->nested_int();
  return (
      c.has_value() && lhs->nested_int() == *c &&
      lhs->nested_int_coeff() == rhs->nested_int_coeff());
}
bool _ge(const char* op, c10::SymNodeImpl* lhs, c10::SymNodeImpl* rhs) {
  if (auto mb_si = lhs->nested_int()) {
    if (auto mb_si2 = rhs->nested_int()) {
      if (*mb_si == *mb_si2) {
        return lhs->nested_int_coeff() >= rhs->nested_int_coeff();
      }
      TORCH_CHECK(false, "nested int ", op, ": Relation is indeterminate");
    }
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (rhs->constant_int() && *rhs->constant_int() <= 2) {
      return true;
    }
    TORCH_CHECK(false, "nested int ", op, ": Relation is indeterminate");
  } else if (rhs->nested_int()) {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (lhs->constant_int() && *lhs->constant_int() < 2) {
      return false;
    }
    TORCH_CHECK(false, "nested int ", op, ": Relation is indeterminate");
  }
  TORCH_INTERNAL_ASSERT(false, "expect at least one nested int");
}
} // namespace

c10::SymNode NestedIntSymNodeImpl::eq(const c10::SymNode& other) {
  return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(
      _eq("eq", this, other.get())));
}

c10::SymNode NestedIntSymNodeImpl::ne(const c10::SymNode& other) {
  return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(
      !_eq("ne", this, other.get())));
}

c10::SymNode NestedIntSymNodeImpl::ge(const c10::SymNode& other) {
  return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(
      _ge("ge", this, other.get())));
}

c10::SymNode NestedIntSymNodeImpl::gt(const c10::SymNode& other) {
  return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(
      !_ge("gt", other.get(), this)));
}

c10::SymNode NestedIntSymNodeImpl::lt(const c10::SymNode& other) {
  return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(
      !_ge("lt", this, other.get())));
}

c10::SymNode NestedIntSymNodeImpl::le(const c10::SymNode& other) {
  return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(
      _ge("le", other.get(), this)));
}

c10::SymNode NestedIntSymNodeImpl::mul(const c10::SymNode& other) {
  if (auto mb_si = other->nested_int()) {
    TORCH_CHECK(false, "nested int cannot be multiplied by nested int");
  }
  c10::optional<int64_t> c = other->constant_int();
  TORCH_CHECK(c.has_value());
  return SymNode(c10::make_intrusive<NestedIntSymNodeImpl>(val_, coeff_ * *c));
}

} // namespace c10
