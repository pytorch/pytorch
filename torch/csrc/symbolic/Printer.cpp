#include <torch/csrc/symbolic/Expr.h>

#include <c10/util/StringUtil.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace torch::symbolic {

namespace {

// sympy.printing.precedence.PRECEDENCE.
enum Precedence : int {
  kOr = 20,
  kAnd = 30,
  kRelational = 35,
  kBitwiseOr = 36,
  kBitwiseXor = 37,
  kBitwiseAnd = 38,
  kAdd = 40,
  kMul = 50,
  kPow = 60,
  kFunc = 70,
  kNot = 100,
  kAtom = 1000,
};

const char* rel_op(Kind k) {
  switch (k) {
    case Kind::Lt:
      return "<";
    case Kind::Le:
      return "<=";
    case Kind::Gt:
      return ">";
    default:
      return ">=";
  }
}

// mlib.to_str(_mpf_, 15, strip_zeros=strip) of a Float, which is
// _print_Float's result for precision 53.
std::string print_float(double v, bool strip) {
  if (v == 0) {
    return "0.0";
  }
  constexpr int dps = 15;
  // to_digits_exp(v, dps + 3): the digits of floor(|v| * 10**fixdps), with
  // bitprec = int(18 * log2(10)) + 10 = 69.
  int exp_bc = 0;
  std::frexp(v, &exp_bc);
  int fixprec = std::max(69 - exp_bc, 0);
  auto fixdps =
      static_cast<int>(fixprec / (std::log(10.0) / std::log(2.0)) + 0.5);
  // A double has at most 1074 fractional digits, so this is exact.
  std::string exact(1500, '\0');
  exact.resize(
      std::snprintf(exact.data(), exact.size(), "%.1074f", std::abs(v)));
  size_t point = exact.find('.');
  std::string digits = exact.substr(0, point) + exact.substr(point + 1, fixdps);
  digits.erase(0, digits.find_first_not_of('0'));
  int exponent = static_cast<int>(digits.size()) - fixdps - 1;
  if (static_cast<int>(digits.size()) > dps && digits[dps] >= '5') {
    digits.resize(dps);
    int i = dps - 1;
    while (i >= 0 && digits[i] == '9') {
      --i;
    }
    if (i >= 0) {
      ++digits[i];
      std::fill(digits.begin() + i + 1, digits.end(), '0');
    } else {
      digits = "1" + std::string(dps - 1, '0');
      ++exponent;
    }
  } else if (static_cast<int>(digits.size()) > dps) {
    digits.resize(dps);
  }
  int split = 1;
  // min_fixed = -5, max_fixed = dps.
  if (-5 < exponent && exponent < dps) {
    if (exponent < 0) {
      digits = std::string(-exponent, '0') + digits;
    } else {
      split = exponent + 1;
      if (split > dps) {
        digits += std::string(split - dps, '0');
      }
    }
    exponent = 0;
  }
  digits = digits.substr(0, split) + "." + digits.substr(split);
  if (strip) {
    digits.erase(digits.find_last_not_of('0') + 1);
    if (digits.back() == '.') {
      digits += '0';
    }
  }
  std::string sign = v < 0 ? "-" : "";
  if (exponent == 0) {
    return sign + digits;
  }
  return sign + digits + (exponent > 0 ? "e+" : "e") + std::to_string(exponent);
}

bool is_negative(const Expr* c) {
  return c->kind == Kind::Float ? c->float_value() < 0
                                : c->is_rational() && c->p < 0;
}

} // namespace

// sympy.printing.str.StrPrinter with the default settings.
class StrPrinter {
 public:
  explicit StrPrinter(ExprArena& arena) : arena_(arena) {}

  std::string print(const Expr* e) {
    ++print_level_;
    std::string s = print_node(e);
    --print_level_;
    return s;
  }

 private:
  std::string print_node(const Expr* e) {
    switch (e->kind) {
      case Kind::Integer:
        return std::to_string(e->p);
      case Kind::Rational:
        return std::to_string(e->p) + "/" + std::to_string(e->q);
      case Kind::Float:
        // full_prec="auto" strips zeros below the top level.
        return print_float(e->float_value(), print_level_ > 1);
      case Kind::IntInfinity:
        return "int_oo";
      case Kind::NegativeIntInfinity:
        return "-int_oo";
      case Kind::Infinity:
        return "oo";
      case Kind::NegativeInfinity:
        return "-oo";
      case Kind::Symbol: {
        const SymbolInfo& info = arena_.symbol_info(e);
        if (info.dummy_index == 0) {
          return info.name;
        }
        if (e == arena_.eps_) {
          // sympy's _eps is an unnamed Dummy, so its name has the global
          // Dummy counter in it.
          throw NativeUnsupported("str of _eps");
        }
        return "_" + info.name;
      }
      case Kind::Pow:
        return print_pow(e);
      case Kind::Mul:
        return print_mul(e);
      case Kind::Add:
        return print_add(e);
      case Kind::BooleanTrue:
        return "True";
      case Kind::BooleanFalse:
        return "False";
      case Kind::Eq:
      case Kind::Ne:
        return std::string(e->kind == Kind::Eq ? "Eq(" : "Ne(") +
            print(e->args[0]) + ", " + print(e->args[1]) + ")";
      case Kind::Lt:
      case Kind::Le:
      case Kind::Gt:
      case Kind::Ge:
        return parenthesize(e->args[0], kRelational) + " " + rel_op(e->kind) +
            " " + parenthesize(e->args[1], kRelational);
      case Kind::Not:
        return "~" + parenthesize(e->args[0], kNot);
      case Kind::And:
        // _print_And first moves relationals whose canonical rhs is -oo to the
        // front; the arena has no -oo.
        return stringify(e->args, " & ", kBitwiseAnd);
      case Kind::Or:
        return stringify(e->args, " | ", kBitwiseOr);
      case Kind::Mod:
      case Kind::PythonMod:
      case Kind::PowByNatural:
      case Kind::FloatPow:
      case Kind::FloatTrueDiv:
      case Kind::IntTrueDiv:
      case Kind::CeilToInt:
      case Kind::FloorToInt:
      case Kind::TruncToInt:
      case Kind::RoundToInt:
      case Kind::RoundDecimal:
      case Kind::ToFloat:
      case Kind::TruncToFloat:
      case Kind::IsNonOverlappingAndDenseIndicator:
      case Kind::ModularIndexing:
      case Kind::BitwiseAnd:
      case Kind::BitwiseOr:
      case Kind::BitwiseXor:
        // _print_Function.
        return std::string(function_name(e->kind)) + "(" +
            stringify(e->args, ", ", 0) + ")";
      case Kind::Max:
      case Kind::Min: {
        // _print_LatticeOp.
        std::vector<const Expr*> args(e->args.begin(), e->args.end());
        std::stable_sort(args.begin(), args.end(), [&](auto a, auto b) {
          return compare_keys(*arena_.sort_key(a), *arena_.sort_key(b)) < 0;
        });
        return std::string(function_name(e->kind)) + "(" +
            stringify(args, ", ", 0) + ")";
      }
      case Kind::FloorDiv:
      case Kind::CleanDiv:
        // FloorDiv._sympystr: parenthesize(arg, PRECEDENCE["Atom"] - 0.5).
        return "(" + parenthesize(e->args[0], kAtom - 1) + "//" +
            parenthesize(e->args[1], kAtom - 1) + ")";
    }
    throw NativeUnsupported("str of an unknown kind");
  }

  // sympy.printing.precedence.precedence.
  int precedence(const Expr* e) {
    switch (e->kind) {
      case Kind::Integer:
        return e->p < 0 ? kAdd : kAtom;
      case Kind::Rational:
        return e->p < 0 ? kAdd : kMul;
      case Kind::Float:
        return is_negative(e) ? kAdd : kAtom;
      case Kind::NegativeInfinity:
        // PRECEDENCE_VALUES["NegativeInfinity"].
        return kAdd;
      case Kind::Mul:
        for (const Expr* a : e->args) {
          if (a->is_function() && precedence(a) < kMul) {
            return kMul;
          }
        }
        return arena_.could_extract_minus_sign(e) ? kAdd : kMul;
      case Kind::Add:
        return kAdd;
      case Kind::Pow:
        return kPow;
      case Kind::Eq:
      case Kind::Ne:
        return kMul;
      case Kind::Lt:
      case Kind::Le:
      case Kind::Gt:
      case Kind::Ge:
        return kRelational;
      case Kind::Not:
        return kNot;
      case Kind::And:
        return kAnd;
      case Kind::Or:
        return kOr;
      case Kind::Mod:
      case Kind::PythonMod:
      case Kind::FloorDiv:
      case Kind::CleanDiv:
      case Kind::FloatTrueDiv:
      case Kind::IntTrueDiv:
      case Kind::ModularIndexing:
        // The classes' precedence attribute.
        return 35;
      case Kind::BitwiseAnd:
        return kBitwiseAnd;
      case Kind::BitwiseOr:
        return kBitwiseOr;
      case Kind::BitwiseXor:
        return kBitwiseXor;
      case Kind::PowByNatural:
        return kMul;
      case Kind::FloatPow:
        return kPow;
      case Kind::CeilToInt:
      case Kind::FloorToInt:
      case Kind::TruncToInt:
      case Kind::RoundToInt:
      case Kind::RoundDecimal:
      case Kind::ToFloat:
      case Kind::TruncToFloat:
      case Kind::IsNonOverlappingAndDenseIndicator:
        // PRECEDENCE_VALUES["Function"].
        return kFunc;
      default:
        return kAtom;
    }
  }

  std::string parenthesize(const Expr* e, int level) {
    std::string s = print(e);
    return precedence(e) <= level ? "(" + s + ")" : s;
  }

  std::string stringify(
      c10::ArrayRef<const Expr*> args,
      const char* sep,
      int level) {
    std::vector<std::string> items;
    for (const Expr* a : args) {
      items.push_back(parenthesize(a, level));
    }
    return c10::Join(sep, items);
  }

  std::string print_add(const Expr* e) {
    std::string r;
    bool first = true;
    for (const Expr* term : arena_.as_ordered_terms(e)) {
      std::string t = print(term);
      char sign = '+';
      if (!t.empty() && t[0] == '-' && term->kind != Kind::Add) {
        sign = '-';
        t.erase(0, 1);
      }
      if (precedence(term) < kAdd || term->kind == Kind::Add) {
        t = "(" + t + ")";
      }
      if (first) {
        r = sign == '-' ? "-" + t : t;
        first = false;
      } else {
        r += std::string(" ") + sign + " " + t;
      }
    }
    return r;
  }

  // The unevaluated-Mul branch of _print_Mul is unreachable: an arena Mul has
  // at most one Number, first and not 1, and no Pow of Integers.
  std::string print_mul(const Expr* e) {
    int prec = precedence(e);
    auto [c, rest] = arena_.as_coeff_Mul(e);
    std::string sign;
    const Expr* expr = e;
    if (is_negative(c)) {
      expr = arena_.keep_coeff(arena_.neg(c), rest);
      sign = "-";
    }
    std::vector<const Expr*> a;
    std::vector<const Expr*> b;
    for (const Expr* item : arena_.as_ordered_factors(expr)) {
      if (item->kind == Kind::Pow && item->args[1]->p < 0) {
        // Arena Pow exponents are Integers.
        const Expr* base = item->args[0];
        const Expr* exp = item->args[1];
        b.push_back(exp->p == -1 ? base : arena_.pow(base, arena_.neg(exp)));
      } else if (item->is_rational()) {
        if (item->p != 1) {
          a.push_back(arena_.integer(item->p));
        }
        if (item->q != 1) {
          b.push_back(arena_.integer(item->q));
        }
      } else {
        a.push_back(item);
      }
    }
    if (a.empty()) {
      a.push_back(arena_.integer(1));
    }
    std::string num = sign + stringify(a, "*", prec);
    if (b.empty()) {
      return num;
    }
    if (b.size() == 1) {
      return num + "/" + parenthesize(b[0], prec);
    }
    return num + "/(" + stringify(b, "*", prec) + ")";
  }

  std::string print_pow(const Expr* e) {
    const Expr* base = e->args[0];
    const Expr* exp = e->args[1];
    if (exp->kind == Kind::Integer && exp->p == -1) {
      return "1/" + parenthesize(base, kPow);
    }
    return parenthesize(base, kPow) + "**" + parenthesize(exp, kPow);
  }

  ExprArena& arena_;
  // Printer._print_level.
  int print_level_ = 0;
};

std::string ExprArena::str(const Expr* e) {
  return StrPrinter(*this).print(e);
}

} // namespace torch::symbolic
