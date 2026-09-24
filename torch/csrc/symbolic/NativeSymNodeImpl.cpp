#include <torch/csrc/symbolic/NativeSymNodeImpl.h>

#include <torch/csrc/symbolic/PyFallback.h>

#include <algorithm>
#include <limits>

namespace torch::symbolic {

namespace {

NativeSymNodeImpl* as_native(const c10::SymNode& node) {
  return dynamic_cast<NativeSymNodeImpl*>(node.get());
}

c10::SymNode to_python(const c10::SymNode& node) {
  auto* native = as_native(node);
  return native ? materialize(*native) : node;
}

std::vector<c10::SymNode> to_python(c10::ArrayRef<c10::SymNode> nodes) {
  std::vector<c10::SymNode> out;
  out.reserve(nodes.size());
  for (const auto& n : nodes) {
    out.push_back(to_python(n));
  }
  return out;
}

bool is_int_oo(const Expr* e) {
  return e->kind == Kind::IntInfinity || e->kind == Kind::NegativeIntInfinity;
}

// functions._is_symbols_binary_summation.
bool is_symbols_binary_summation(const Expr* e) {
  return e->kind == Kind::Add && e->args.size() == 2 &&
      e->args[0]->kind == Kind::Symbol && e->args[1]->kind == Kind::Symbol &&
      e->args[0] != e->args[1];
}

// The optimized_summation flag of _optimized_add(lhs, rhs, lopt, ropt), whose
// expression is always Add(lhs, rhs) = out. The args of an optimized summation
// are distinct symbols sorted by Basic.compare, so _binary_search_insert_arg
// fails exactly when the arg is already there.
bool optimized_add_flag(
    ExprArena& arena,
    const Expr* lhs,
    bool lopt,
    const Expr* rhs,
    bool ropt,
    const Expr* out) {
  lopt = lopt || is_symbols_binary_summation(lhs);
  ropt = ropt || is_symbols_binary_summation(rhs);
  auto contains = [](const Expr* sum, const Expr* a) {
    return std::find(sum->args.begin(), sum->args.end(), a) != sum->args.end();
  };
  if (lopt && ropt) {
    if (arena.compare(lhs->args.back(), rhs->args.front()) < 0 ||
        arena.compare(lhs->args.front(), rhs->args.back()) > 0) {
      return true;
    }
    if (lhs->args.size() <= 2 && rhs->args.size() <= 2 &&
        std::none_of(rhs->args.begin(), rhs->args.end(), [&](const Expr* a) {
          return contains(lhs, a);
        })) {
      return true;
    }
  }
  if (lopt && rhs->kind == Kind::Symbol && !contains(lhs, rhs)) {
    return true;
  }
  if (ropt && lhs->kind == Kind::Symbol && !contains(rhs, lhs)) {
    return true;
  }
  return is_symbols_binary_summation(out);
}

// Python int power; nullopt on a negative exponent (a float) or overflow.
std::optional<int64_t> int_pow(int64_t base, int64_t exp) {
  if (exp < 0) {
    return std::nullopt;
  }
  int64_t r = 1;
  while (exp != 0) {
    if ((exp & 1) && __builtin_mul_overflow(r, base, &r)) {
      return std::nullopt;
    }
    exp >>= 1;
    if (exp != 0 && __builtin_mul_overflow(base, base, &base)) {
      return std::nullopt;
    }
  }
  return r;
}

// bool() of an evaluate_expr result; nullopt for no result.
std::optional<bool> as_bool(const Expr* e) {
  if (e == nullptr) {
    return std::nullopt;
  }
  switch (e->kind) {
    case Kind::BooleanTrue:
      return true;
    case Kind::BooleanFalse:
      return false;
    case Kind::Integer:
      return e->p != 0;
    default:
      return std::nullopt;
  }
}

} // namespace

std::optional<int64_t> NativeSymNodeImpl::maybe_as_int() {
  {
    auto lock = lock_env(*env_);
    if (env_->replacements_empty()) {
      switch (expr_->kind) {
        case Kind::Integer:
          return expr_->p;
        case Kind::Rational:
          return expr_->p / expr_->q;
        case Kind::IntInfinity:
        case Kind::NegativeIntInfinity:
          break;
        default:
          // Not a number: a native expression without free symbols is one.
          return std::nullopt;
      }
    }
  }
  return materialize(*this)->maybe_as_int();
}

std::string NativeSymNodeImpl::str() {
  {
    auto lock = lock_env(*env_);
    if (env_->replacements_empty()) {
      return env_->arena().str(expr_);
    }
  }
  return materialize(*this)->str();
}

c10::SymNode NativeSymNodeImpl::wrap_int(int64_t num) {
  auto lock = lock_env(*env_);
  return c10::make_intrusive<NativeSymNodeImpl>(
      env_, env_->arena().integer(num), PyType::Int, num, num);
}

c10::SymNode NativeSymNodeImpl::wrap_float(double num) {
  return materialize(*this)->wrap_float(num);
}

c10::SymNode NativeSymNodeImpl::wrap_bool(bool num) {
  return c10::make_intrusive<NativeSymNodeImpl>(
      env_, env_->arena().boolean(num), PyType::Bool, num, num);
}

c10::SymNode NativeSymNodeImpl::binary(
    Op op,
    BinaryFn fallback,
    const c10::SymNode& other) {
  auto* o = as_native(other);
  if (o != nullptr && o->env_ == env_) {
    auto lock = lock_env(*env_);
    try {
      if (auto r = try_binary(op, *o)) {
        return r;
      }
    } catch (const NativeUnsupported&) {
    }
  }
  return (materialize(*this).get()->*fallback)(to_python(other));
}

c10::SymNode NativeSymNodeImpl::try_binary(
    Op op,
    const NativeSymNodeImpl& other) {
  bool logical = op == Op::And || op == Op::Or;
  PyType operand = logical ? PyType::Bool : PyType::Int;
  if (!env_->replacements_empty() || pytype_ != operand ||
      other.pytype_ != operand) {
    return {};
  }

  // The hint first, as Python computes it; a hint that Python cannot
  // represent natively or that raises goes to the fallback.
  Hint hint;
  if (has_hint() && !std::holds_alternative<std::monostate>(other.hint_)) {
    if (logical) {
      bool x = std::get<bool>(hint_);
      bool y = std::get<bool>(other.hint_);
      hint = op == Op::And ? (x && y) : (x || y);
    } else {
      int64_t x = std::get<int64_t>(hint_);
      int64_t y = std::get<int64_t>(other.hint_);
      int64_t r = 0;
      switch (op) {
        case Op::Add:
          if (__builtin_add_overflow(x, y, &r)) {
            return {};
          }
          hint = r;
          break;
        case Op::Sub:
          if (__builtin_sub_overflow(x, y, &r)) {
            return {};
          }
          hint = r;
          break;
        case Op::Mul:
          if (__builtin_mul_overflow(x, y, &r)) {
            return {};
          }
          hint = r;
          break;
        case Op::FloorDiv:
          if (y == 0 || (x == std::numeric_limits<int64_t>::min() && y == -1)) {
            return {};
          }
          r = x / y;
          if (x % y != 0 && ((x < 0) != (y < 0))) {
            --r;
          }
          hint = r;
          break;
        case Op::Mod:
          if (y == 0) {
            return {};
          }
          r = y == -1 ? 0 : x % y;
          if (r != 0 && ((r < 0) != (y < 0))) {
            r += y;
          }
          hint = r;
          break;
        case Op::PowByNatural: {
          auto p = int_pow(x, y);
          if (!p) {
            return {};
          }
          hint = *p;
          break;
        }
        case Op::Min:
          hint = std::min(x, y);
          break;
        case Op::Max:
          hint = std::max(x, y);
          break;
        case Op::Eq:
          hint = x == y;
          break;
        case Op::Ne:
          hint = x != y;
          break;
        case Op::Gt:
          hint = x > y;
          break;
        case Op::Lt:
          hint = x < y;
          break;
        case Op::Le:
          hint = x <= y;
          break;
        case Op::Ge:
          hint = x >= y;
          break;
        case Op::And:
        case Op::Or:
          break;
      }
    }
  }

  ExprArena& arena = env_->arena();
  const Expr* a = expr_;
  const Expr* b = other.expr_;
  // IntInfinity overloads the Python operators.
  if ((op == Op::Add || op == Op::Sub || op == Op::Mul) &&
      (is_int_oo(a) || is_int_oo(b))) {
    return {};
  }
  const Expr* out = nullptr;
  bool optimized_summation = false;
  switch (op) {
    case Op::Add:
      out = arena.add({a, b});
      optimized_summation = optimized_add_flag(
          arena, a, optimized_summation_, b, other.optimized_summation_, out);
      break;
    case Op::Sub:
      out = arena.sub(a, b);
      break;
    case Op::Mul:
      out = arena.mul({a, b});
      break;
    case Op::FloorDiv:
      out = arena.function(Kind::FloorDiv, {a, b});
      break;
    case Op::Mod: {
      // Range-dependent: the choice must match what _symop_cache would
      // return, which holds only while the env is pristine.
      auto nonnegative = [&](const Expr* e) {
        return arena.ask(e, Fact::nonnegative) == Tri::True ||
            env_->bound_lower_nonnegative(e);
      };
      bool mod = nonnegative(a) && nonnegative(b);
      out = arena.function(mod ? Kind::Mod : Kind::PythonMod, {a, b});
      break;
    }
    case Op::PowByNatural:
      out = arena.function(Kind::PowByNatural, {a, b});
      break;
    case Op::Min:
      out = arena.function(Kind::Min, {a, b});
      break;
    case Op::Max:
      out = arena.function(Kind::Max, {a, b});
      break;
    case Op::Eq:
    case Op::Ne:
    case Op::Gt:
    case Op::Lt:
    case Op::Le:
    case Op::Ge: {
      auto unbacked = [&](const Expr* e) {
        if (e->kind != Kind::Symbol) {
          return false;
        }
        const std::string& name = arena.symbol_info(e).name;
        return !name.empty() && (name[0] == 'u' || name[0] == 'U');
      };
      bool evaluate =
          !((unbacked(a) && b->is_number()) || (unbacked(b) && a->is_number()));
      Kind kind = op == Op::Eq ? Kind::Eq
          : op == Op::Ne       ? Kind::Ne
          : op == Op::Gt       ? Kind::Gt
          : op == Op::Lt       ? Kind::Lt
          : op == Op::Le       ? Kind::Le
                               : Kind::Ge;
      out = arena.rel(kind, a, b, evaluate);
      break;
    }
    case Op::And:
      out = arena.logical_and({a, b});
      break;
    case Op::Or:
      out = arena.logical_or({a, b});
      break;
  }
  bool int_result = op <= Op::Max;
  return c10::make_intrusive<NativeSymNodeImpl>(
      env_,
      out,
      int_result ? PyType::Int : PyType::Bool,
      hint,
      Hint{},
      optimized_summation);
}

c10::SymNode NativeSymNodeImpl::unary(bool is_not, UnaryFn fallback) {
  {
    auto lock = lock_env(*env_);
    try {
      if (auto r = try_unary(is_not)) {
        return r;
      }
    } catch (const NativeUnsupported&) {
    }
  }
  return (materialize(*this).get()->*fallback)();
}

c10::SymNode NativeSymNodeImpl::try_unary(bool is_not) {
  PyType operand = is_not ? PyType::Bool : PyType::Int;
  if (!env_->replacements_empty() || pytype_ != operand) {
    return {};
  }
  ExprArena& arena = env_->arena();
  Hint hint;
  const Expr* out = nullptr;
  if (is_not) {
    if (has_hint()) {
      hint = !std::get<bool>(hint_);
    }
    out = arena.logical_not(expr_);
  } else {
    if (has_hint()) {
      int64_t x = std::get<int64_t>(hint_);
      if (x == std::numeric_limits<int64_t>::min()) {
        return {};
      }
      hint = -x;
    }
    if (is_int_oo(expr_)) {
      return {};
    }
    out = arena.neg(expr_);
  }
  return c10::make_intrusive<NativeSymNodeImpl>(env_, out, operand, hint);
}

#define NATIVE_BINARY(name, op)                                     \
  c10::SymNode NativeSymNodeImpl::name(const c10::SymNode& other) { \
    return binary(Op::op, &c10::SymNodeImpl::name, other);          \
  }
NATIVE_BINARY(add, Add)
NATIVE_BINARY(sub, Sub)
NATIVE_BINARY(mul, Mul)
NATIVE_BINARY(floordiv, FloorDiv)
NATIVE_BINARY(int_floordiv, FloorDiv)
NATIVE_BINARY(mod, Mod)
NATIVE_BINARY(pow_by_natural, PowByNatural)
NATIVE_BINARY(sym_min, Min)
NATIVE_BINARY(sym_max, Max)
NATIVE_BINARY(eq, Eq)
NATIVE_BINARY(ne, Ne)
NATIVE_BINARY(gt, Gt)
NATIVE_BINARY(lt, Lt)
NATIVE_BINARY(le, Le)
NATIVE_BINARY(ge, Ge)
NATIVE_BINARY(sym_and, And)
NATIVE_BINARY(sym_or, Or)
#undef NATIVE_BINARY

c10::SymNode NativeSymNodeImpl::neg() {
  return unary(false, &c10::SymNodeImpl::neg);
}

c10::SymNode NativeSymNodeImpl::sym_not() {
  return unary(true, &c10::SymNodeImpl::sym_not);
}

#define PYTHON_BINARY(name)                                         \
  c10::SymNode NativeSymNodeImpl::name(const c10::SymNode& other) { \
    return materialize(*this)->name(to_python(other));              \
  }
PYTHON_BINARY(truediv)
PYTHON_BINARY(float_truediv)
PYTHON_BINARY(int_truediv)
PYTHON_BINARY(pow)
PYTHON_BINARY(float_pow)
#undef PYTHON_BINARY

#define PYTHON_UNARY(name)                 \
  c10::SymNode NativeSymNodeImpl::name() { \
    return materialize(*this)->name();     \
  }
PYTHON_UNARY(ceil)
PYTHON_UNARY(floor)
PYTHON_UNARY(sym_float)
#undef PYTHON_UNARY

c10::SymNode NativeSymNodeImpl::sym_ite(
    const c10::SymNode& then_val,
    const c10::SymNode& else_val) {
  return materialize(*this)->sym_ite(to_python(then_val), to_python(else_val));
}

#define PYTHON_SIZES_STRIDES(name)                                         \
  c10::SymNode NativeSymNodeImpl::name(                                    \
      c10::ArrayRef<c10::SymNode> sizes,                                   \
      c10::ArrayRef<c10::SymNode> strides) {                               \
    return materialize(*this)->name(to_python(sizes), to_python(strides)); \
  }
PYTHON_SIZES_STRIDES(is_contiguous)
PYTHON_SIZES_STRIDES(is_channels_last_contiguous_2d)
PYTHON_SIZES_STRIDES(is_channels_last_contiguous_3d)
PYTHON_SIZES_STRIDES(is_channels_last_strides_2d)
PYTHON_SIZES_STRIDES(is_channels_last_strides_3d)
PYTHON_SIZES_STRIDES(is_non_overlapping_and_dense)
#undef PYTHON_SIZES_STRIDES

const Expr* NativeSymNodeImpl::evaluate(std::optional<bool> fallback_value) {
  if (!native_config_is_default()) {
    return nullptr;
  }
  auto lock = lock_env(*env_);
  return env_->evaluate_expr(expr_, hint_, fallback_value).value_or(nullptr);
}

int64_t NativeSymNodeImpl::guard_int(const char* file, int64_t line) {
  const Expr* r = evaluate(std::nullopt);
  if (r != nullptr && r->kind == Kind::Integer) {
    return r->p;
  }
  return materialize(*this)->guard_int(file, line);
}

bool NativeSymNodeImpl::guard_bool(const char* file, int64_t line) {
  if (auto r = as_bool(evaluate(std::nullopt))) {
    return *r;
  }
  return materialize(*this)->guard_bool(file, line);
}

bool NativeSymNodeImpl::guard_or_false(const char* file, int64_t line) {
  if (pytype_ == PyType::Bool) {
    if (auto r = as_bool(evaluate(false))) {
      return *r;
    }
  }
  return materialize(*this)->guard_or_false(file, line);
}

bool NativeSymNodeImpl::guard_or_true(const char* file, int64_t line) {
  if (pytype_ == PyType::Bool) {
    if (auto r = as_bool(evaluate(true))) {
      return *r;
    }
  }
  return materialize(*this)->guard_or_true(file, line);
}

bool NativeSymNodeImpl::statically_known_true(const char* file, int64_t line) {
  if (pytype_ == PyType::Bool) {
    // _sym_node_hint_disproves
    if (hint_ == Hint(false)) {
      return false;
    }
    std::optional<const Expr*> r;
    {
      auto lock = lock_env(*env_);
      r = env_->static_eval(expr_);
    }
    if (r && *r == nullptr) {
      return false;
    }
    if (auto b = r ? as_bool(*r) : std::nullopt) {
      return *b;
    }
  }
  return materialize(*this)->statically_known_true(file, line);
}

bool NativeSymNodeImpl::expect_true(const char* file, int64_t line) {
  // A native env has prefer_deferred_runtime_asserts_over_guards unset, and
  // evaluate() answers only for mirrored (backed) symbols.
  if (has_hint()) {
    if (auto r = as_bool(evaluate(std::nullopt))) {
      return *r;
    }
  }
  return materialize(*this)->expect_true(file, line);
}

double NativeSymNodeImpl::guard_float(const char* file, int64_t line) {
  return materialize(*this)->guard_float(file, line);
}

bool NativeSymNodeImpl::guard_size_oblivious(const char* file, int64_t line) {
  return materialize(*this)->guard_size_oblivious(file, line);
}

int64_t NativeSymNodeImpl::int_() {
  return guard_int("", 0);
}

bool NativeSymNodeImpl::bool_() {
  return guard_bool("", 0);
}

} // namespace torch::symbolic
