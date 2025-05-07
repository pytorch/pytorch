
#pragma once

#include <c10/macros/Export.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-parameter")

namespace c10 {

class SymNodeImpl;
using SymNode = c10::intrusive_ptr<SymNodeImpl>;

// When you add a method, you also need to edit
// torch/csrc/jit/python/init.cpp
// torch/csrc/utils/python_symnode.h
// c10/core/ConstantSymNodeImpl.h
class C10_API SymNodeImpl : public c10::intrusive_ptr_target {
 public:
  ~SymNodeImpl() override = default;

  template <typename T>
  c10::intrusive_ptr<T> dyn_cast() const {
    return c10::intrusive_ptr<T>::reclaim_copy(dynamic_cast<T*>(this));
  }

  // these could be pure virtual when we implement LTC versions
  virtual bool is_int() {
    TORCH_CHECK(false, "NYI");
  }
  virtual bool is_bool() {
    TORCH_CHECK(false, "NYI");
  }
  virtual bool is_float() {
    TORCH_CHECK(false, "NYI");
  }
  virtual bool is_nested_int() const {
    return false;
  }
  virtual SymNode add(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode sub(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode mul(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  // NB: legacy, prefer float_truediv or int_truediv
  virtual SymNode truediv(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode float_truediv(const SymNode& other) {
    return truediv(other);
  }
  virtual SymNode int_truediv(const SymNode& other) {
    return truediv(other);
  }
  // NB: legacy, prefer float_pow or pow_by_natural
  virtual SymNode pow(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode float_pow(const SymNode& other) {
    return pow(other);
  }
  virtual SymNode pow_by_natural(const SymNode& other) {
    return pow(other);
  }
  // NB: legacy, prefer int_floordiv
  virtual SymNode floordiv(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode int_floordiv(const SymNode& other) {
    return floordiv(other);
  }
  virtual SymNode mod(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode eq(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode ne(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode gt(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode lt(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode le(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode ge(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode ceil() {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode floor() {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode neg() {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode sym_min(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode sym_max(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode sym_or(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode sym_and(const SymNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode sym_not() {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode sym_ite(const SymNode& then_val, const SymNode& else_val) {
    TORCH_CHECK(false, "NYI");
  }
  // NB: self is ignored here, only the arguments are used
  virtual SymNode is_contiguous(
      ArrayRef<SymNode> sizes,
      ArrayRef<SymNode> strides) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode is_channels_last_contiguous_2d(
      ArrayRef<SymNode> sizes,
      ArrayRef<SymNode> strides) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode is_channels_last_contiguous_3d(
      ArrayRef<SymNode> sizes,
      ArrayRef<SymNode> strides) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode is_channels_last_strides_2d(
      ArrayRef<SymNode> sizes,
      ArrayRef<SymNode> strides) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode is_channels_last_strides_3d(
      ArrayRef<SymNode> sizes,
      ArrayRef<SymNode> strides) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode is_non_overlapping_and_dense(
      ArrayRef<SymNode> sizes,
      ArrayRef<SymNode> strides) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode clone() {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode sym_float() {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode wrap_int(int64_t num) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode wrap_float(double num) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymNode wrap_bool(bool num) {
    TORCH_CHECK(false, "NYI");
  }
  virtual int64_t guard_int(const char* file, int64_t line) {
    TORCH_CHECK(false, "NYI");
  }
  virtual bool guard_bool(const char* file, int64_t line) {
    TORCH_CHECK(false, "NYI");
  }
  virtual double guard_float(const char* file, int64_t line) {
    TORCH_CHECK(false, "NYI");
  }
  virtual bool guard_size_oblivious(const char* file, int64_t line) {
    // No improvement for unbacked SymBools by default, replace this
    // with a better implementation!
    return guard_bool(file, line);
  }
  virtual bool guard_or_false(const char* file, int64_t line) {
    // No improvement for unbacked SymBools by default, replace this
    // with a better implementation!
    return guard_bool(file, line);
  }
  virtual bool statically_known_true(const char* file, int64_t line) {
    // No improvement for unbacked SymBools by default, replace this
    // with a better implementation!
    return guard_bool(file, line);
  }
  virtual bool guard_or_true(const char* file, int64_t line) {
    // No improvement for unbacked SymBools by default, replace this
    // with a better implementation!
    return guard_bool(file, line);
  }
  virtual bool expect_true(const char* file, int64_t line) {
    // No improvement for unbacked SymBools by default, replace this
    // with a better implementation!
    return guard_bool(file, line);
  }
  virtual bool expect_size(const char* file, int64_t line) {
    // No improvement for unbacked SymInts by default, replace this
    // with a better implementation!
    return ge(wrap_int(0))->guard_bool(file, line);
  }
  virtual int64_t int_() {
    TORCH_CHECK(false, "NYI");
  }
  virtual bool bool_() {
    TORCH_CHECK(false, "NYI");
  }
  virtual bool has_hint() {
    TORCH_CHECK(false, "NYI");
  }
  virtual std::string str() {
    TORCH_CHECK(false, "NYI");
  }
  virtual std::string _graph_repr() {
    return str();
  }
  virtual std::optional<int64_t> nested_int() {
    return std::nullopt;
  }
  virtual std::optional<int64_t> nested_int_coeff() {
    return std::nullopt;
  }
  virtual std::optional<int64_t> constant_int() {
    return std::nullopt;
  }
  virtual std::optional<bool> constant_bool() {
    return std::nullopt;
  }
  virtual std::optional<int64_t> maybe_as_int() {
    return std::nullopt;
  }
  virtual bool is_constant() {
    return false;
  }
  virtual bool is_symbolic() {
    return true;
  }
  std::ostream& operator<<(std::ostream& os) {
    os << str();
    return os;
  }
};

} // namespace c10
C10_DIAGNOSTIC_POP()
