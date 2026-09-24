#pragma once

#include <c10/core/SymNodeImpl.h>
#include <torch/csrc/symbolic/NativeShapeEnv.h>

// Port of torch.fx.experimental.sym_node.SymNode for int and bool nodes. Nodes
// are immutable. Whatever is not ported calls the same method on the Python
// SymNode the node stands for (PyFallback.h).

namespace torch::symbolic {

enum class PyType : uint8_t { Int, Bool };

class NativeSymNodeImpl final : public c10::SymNodeImpl {
 public:
  // Types are as for SymNode: a hint of the pytype or None, and a constant
  // only from wrap_*.
  NativeSymNodeImpl(
      c10::intrusive_ptr<NativeShapeEnv> env,
      const Expr* expr,
      PyType pytype,
      Hint hint,
      Hint constant = {},
      bool optimized_summation = false)
      : env_(std::move(env)),
        expr_(expr),
        pytype_(pytype),
        hint_(hint),
        constant_(constant),
        optimized_summation_(optimized_summation) {}

  const c10::intrusive_ptr<NativeShapeEnv>& env() const {
    return env_;
  }
  // SymNode._expr: before replacements.
  const Expr* expr() const {
    return expr_;
  }
  PyType pytype() const {
    return pytype_;
  }
  const Hint& hint() const {
    return hint_;
  }
  const Hint& constant() const {
    return constant_;
  }
  bool optimized_summation() const {
    return optimized_summation_;
  }

  bool is_int() override {
    return pytype_ == PyType::Int;
  }
  bool is_bool() override {
    return pytype_ == PyType::Bool;
  }
  bool is_float() override {
    return false;
  }
  bool is_nested_int() const override {
    return false;
  }
  bool has_hint() override {
    return !std::holds_alternative<std::monostate>(hint_);
  }
  std::optional<int64_t> maybe_as_int() override;
  c10::SymNode clone() override {
    return c10::intrusive_ptr<c10::SymNodeImpl>::reclaim_copy(this);
  }
  std::string str() override;
  std::string _graph_repr() override {
    return str();
  }

  c10::SymNode wrap_int(int64_t num) override;
  c10::SymNode wrap_float(double num) override;
  c10::SymNode wrap_bool(bool num) override;

  c10::SymNode add(const c10::SymNode& other) override;
  c10::SymNode sub(const c10::SymNode& other) override;
  c10::SymNode mul(const c10::SymNode& other) override;
  c10::SymNode truediv(const c10::SymNode& other) override;
  c10::SymNode float_truediv(const c10::SymNode& other) override;
  c10::SymNode int_truediv(const c10::SymNode& other) override;
  c10::SymNode pow(const c10::SymNode& other) override;
  c10::SymNode float_pow(const c10::SymNode& other) override;
  c10::SymNode pow_by_natural(const c10::SymNode& other) override;
  c10::SymNode floordiv(const c10::SymNode& other) override;
  c10::SymNode int_floordiv(const c10::SymNode& other) override;
  c10::SymNode mod(const c10::SymNode& other) override;
  c10::SymNode eq(const c10::SymNode& other) override;
  c10::SymNode ne(const c10::SymNode& other) override;
  c10::SymNode gt(const c10::SymNode& other) override;
  c10::SymNode lt(const c10::SymNode& other) override;
  c10::SymNode le(const c10::SymNode& other) override;
  c10::SymNode ge(const c10::SymNode& other) override;
  c10::SymNode sym_min(const c10::SymNode& other) override;
  c10::SymNode sym_max(const c10::SymNode& other) override;
  c10::SymNode sym_or(const c10::SymNode& other) override;
  c10::SymNode sym_and(const c10::SymNode& other) override;
  c10::SymNode ceil() override;
  c10::SymNode floor() override;
  c10::SymNode neg() override;
  c10::SymNode sym_not() override;
  c10::SymNode sym_float() override;
  c10::SymNode sym_ite(
      const c10::SymNode& then_val,
      const c10::SymNode& else_val) override;

  c10::SymNode is_contiguous(
      c10::ArrayRef<c10::SymNode> sizes,
      c10::ArrayRef<c10::SymNode> strides) override;
  c10::SymNode is_channels_last_contiguous_2d(
      c10::ArrayRef<c10::SymNode> sizes,
      c10::ArrayRef<c10::SymNode> strides) override;
  c10::SymNode is_channels_last_contiguous_3d(
      c10::ArrayRef<c10::SymNode> sizes,
      c10::ArrayRef<c10::SymNode> strides) override;
  c10::SymNode is_channels_last_strides_2d(
      c10::ArrayRef<c10::SymNode> sizes,
      c10::ArrayRef<c10::SymNode> strides) override;
  c10::SymNode is_channels_last_strides_3d(
      c10::ArrayRef<c10::SymNode> sizes,
      c10::ArrayRef<c10::SymNode> strides) override;
  c10::SymNode is_non_overlapping_and_dense(
      c10::ArrayRef<c10::SymNode> sizes,
      c10::ArrayRef<c10::SymNode> strides) override;

  int64_t guard_int(const char* file, int64_t line) override;
  bool guard_bool(const char* file, int64_t line) override;
  double guard_float(const char* file, int64_t line) override;
  bool guard_size_oblivious(const char* file, int64_t line) override;
  bool guard_or_false(const char* file, int64_t line) override;
  bool statically_known_true(const char* file, int64_t line) override;
  bool guard_or_true(const char* file, int64_t line) override;
  bool expect_true(const char* file, int64_t line) override;
  int64_t int_() override;
  bool bool_() override;

 private:
  enum class Op : uint8_t {
    Add,
    Sub,
    Mul,
    FloorDiv,
    Mod,
    PowByNatural,
    Min,
    Max,
    Eq,
    Ne,
    Gt,
    Lt,
    Le,
    Ge,
    And,
    Or,
  };
  using BinaryFn = c10::SymNode (c10::SymNodeImpl::*)(const c10::SymNode&);
  using UnaryFn = c10::SymNode (c10::SymNodeImpl::*)();

  // binary_magic_impl and unary_magic_impl: natively when possible, else
  // `fallback` on the Python SymNode.
  c10::SymNode binary(Op op, BinaryFn fallback, const c10::SymNode& other);
  c10::SymNode unary(bool is_not, UnaryFn fallback);
  // The native result, or null. The caller holds the env lock.
  c10::SymNode try_binary(Op op, const NativeSymNodeImpl& other);
  c10::SymNode try_unary(bool is_not);

  c10::intrusive_ptr<NativeShapeEnv> env_;
  const Expr* expr_;
  PyType pytype_;
  Hint hint_;
  Hint constant_;
  bool optimized_summation_;
};

} // namespace torch::symbolic
