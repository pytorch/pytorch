#include <torch/csrc/symbolic/Expr.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <initializer_list>

namespace torch::symbolic {

namespace {

using i128 = __int128;

// Generated from sympy.core.assumptions._assume_rules (sympy 1.14); bit i is
// Fact(i). test_native_symnode.py checks these against sympy.
// clang-format off
constexpr std::array<std::array<Implication, 2>, kNumFacts> kFullImplications = {{
    {{{0x00000000, 0x0ffffe7e}, {0x00000000, 0x00000000}}}, // commutative
    {{{0x00000000, 0x00f00200}, {0x150000e9, 0x0a000114}}}, // integer
    {{{0x00000000, 0x00000000}, {0x00080041, 0x08f00202}}}, // noninteger
    {{{0x00000000, 0x00f00202}, {0x150000e1, 0x0a000110}}}, // rational
    {{{0x00000000, 0x00000000}, {0x140804e5, 0x08f0030a}}}, // irrational
    {{{0x00000000, 0x00f07e1a}, {0x140000c1, 0x08000100}}}, // real
    {{{0x00000000, 0x00fffe3e}, {0x00000001, 0x08000000}}}, // extended_real
    {{{0x00000100, 0x0ff07e3a}, {0x00000000, 0x00000100}}}, // finite
    {{{0x00000080, 0x00000000}, {0x00000000, 0x0ff07eba}}}, // infinite
    {{{0x00000000, 0x00000000}, {0x151660eb, 0x0ae99d14}}}, // zero
    {{{0x00000000, 0x00c01800}, {0x140800e1, 0x08000300}}}, // nonzero
    {{{0x00000000, 0x00c00000}, {0x140aa4e1, 0x08055300}}}, // positive
    {{{0x00000000, 0x00000000}, {0x140d44e1, 0x08c2ab00}}}, // negative
    {{{0x00000000, 0x00c00a00}, {0x140200e1, 0x08011100}}}, // nonnegative
    {{{0x00000000, 0x00001200}, {0x140400e1, 0x08c08900}}}, // nonpositive
    {{{0x00000000, 0x00c00800}, {0x000a0041, 0x08055200}}}, // extended_positive
    {{{0x00000000, 0x00001000}, {0x000c0041, 0x08c2aa00}}}, // extended_negative
    {{{0x00000000, 0x00c0aa00}, {0x00000041, 0x08011000}}}, // extended_nonnegative
    {{{0x00000000, 0x00015200}, {0x00000041, 0x08c08800}}}, // extended_nonpositive
    {{{0x00000000, 0x00c19c00}, {0x00000041, 0x08000200}}}, // extended_nonzero
    {{{0x00000000, 0x00000200}, {0x150000eb, 0x0a200114}}}, // even
    {{{0x00000000, 0x00000000}, {0x150804eb, 0x0a100314}}}, // odd
    {{{0x00000000, 0x00000000}, {0x150aaceb, 0x0a855314}}}, // prime
    {{{0x00000000, 0x00000000}, {0x150aaceb, 0x0a455314}}}, // composite
    {{{0x00000000, 0x00f0020a}, {0x04000081, 0x02000100}}}, // algebraic
    {{{0x00000000, 0x00000000}, {0x04000081, 0x01f0030a}}}, // transcendental
    {{{0x00000000, 0x0bf07e3a}, {0x00000081, 0x00000100}}}, // complex
    {{{0x00000000, 0x00000000}, {0x24000081, 0x00ffff7e}}}, // imaginary
    {{{0x00000000, 0x00f07e3a}, {0x00000000, 0x00000000}}}, // hermitian
    {{{0x00000000, 0x08000000}, {0x00000000, 0x00000000}}}, // antihermitian
    {{{0x00000000, 0x00000000}, {0x00000000, 0x00000000}}}, // polar
}};
constexpr std::array<BetaRule, 37> kBetaRules = {{
    {0x00100800, 0x00400000, Fact::composite, true},
    {0x00000800, 0x00c00000, Fact::even, false},
    {0x00000002, 0x00200000, Fact::even, true},
    {0x00000040, 0x00008200, Fact::extended_negative, true},
    {0x000c0000, 0x00000000, Fact::extended_negative, true},
    {0x00000040, 0x00010000, Fact::extended_nonnegative, true},
    {0x00000040, 0x00008000, Fact::extended_nonpositive, true},
    {0x00000040, 0x00000200, Fact::extended_nonzero, true},
    {0x00000040, 0x00010200, Fact::extended_positive, true},
    {0x000a0000, 0x00000000, Fact::extended_positive, true},
    {0x00000000, 0x00000120, Fact::extended_real, false},
    {0x00000000, 0x00018200, Fact::extended_real, false},
    {0x00000040, 0x00000020, Fact::infinite, true},
    {0x00000020, 0x00000008, Fact::irrational, true},
    {0x00000020, 0x00000a00, Fact::negative, true},
    {0x00004400, 0x00000000, Fact::negative, true},
    {0x00010080, 0x00000000, Fact::negative, true},
    {0x00000040, 0x00000002, Fact::noninteger, true},
    {0x00000020, 0x00001000, Fact::nonnegative, true},
    {0x00020080, 0x00000000, Fact::nonnegative, true},
    {0x00000020, 0x00000800, Fact::nonpositive, true},
    {0x00040080, 0x00000000, Fact::nonpositive, true},
    {0x00080080, 0x00000000, Fact::nonzero, true},
    {0x00000002, 0x00100000, Fact::odd, true},
    {0x00100000, 0x00c00000, Fact::positive, false},
    {0x00000020, 0x00001200, Fact::positive, true},
    {0x00002400, 0x00000000, Fact::positive, true},
    {0x00008080, 0x00000000, Fact::positive, true},
    {0x00100800, 0x00800000, Fact::prime, true},
    {0x00000000, 0x00001a00, Fact::real, false},
    {0x00000040, 0x00000100, Fact::real, true},
    {0x000000c0, 0x00000000, Fact::real, true},
    {0x04000000, 0x01000000, Fact::transcendental, true},
    {0x00000040, 0x00018000, Fact::zero, true},
    {0x00000020, 0x00001800, Fact::zero, true},
    {0x00060000, 0x00000000, Fact::zero, true},
    {0x00006000, 0x00000000, Fact::zero, true},
}};
constexpr std::array<std::array<uint64_t, 2>, kNumFacts> kBetaTriggers = {{
    {{0x0000000000000000, 0x0000000000000000}}, // commutative
    {{0x0000000022024988, 0x000000060afd41ec}}, // integer
    {{0x0000000000000000, 0x00000000e2405378}}, // noninteger
    {{0x0000000022026988, 0x000000060a7f41e8}}, // rational
    {{0x0000000000000000, 0x000000010e3dc378}}, // irrational
    {{0x0000000000021d88, 0x000000070a7f61e8}}, // real
    {{0x0000000000000000, 0x00000002c00211e8}}, // extended_real
    {{0x0000000000020988, 0x00000000c8690400}}, // finite
    {{0x00000000c8690400, 0x0000000000020988}}, // infinite
    {{0x0000000022004988, 0x0000000000000000}}, // zero
    {{0x0000000420140000, 0x000000010e3fe378}}, // nonzero
    {{0x0000000420104000, 0x0000000110022003}}, // positive
    {{0x0000000422040000, 0x0000000100022000}}, // negative
    {{0x0000000020104988, 0x0000001f0e7223c0}}, // nonnegative
    {{0x0000000022040988, 0x0000001f004fe0b8}}, // nonpositive
    {{0x0000000620104848, 0x00000000ea4e1000}}, // extended_positive
    {{0x0000000622040920, 0x00000000e0735000}}, // extended_negative
    {{0x00000000201048c8, 0x0000000ee20e13c0}}, // extended_nonnegative
    {{0x00000000220409a0, 0x0000000ee03250b8}}, // extended_nonpositive
    {{0x0000000620140860, 0x00000000e2425378}}, // extended_nonzero
    {{0x0000000022804988, 0x000000061b7d41e9}}, // even
    {{0x0000000000000004, 0x000000000e3dc378}}, // odd
    {{0x0000000001000003, 0x0000000000800004}}, // prime
    {{0x0000000011000002, 0x0000000000800004}}, // composite
    {{0x0000000122026988, 0x00000000c8690400}}, // algebraic
    {{0x0000000000000000, 0x00000000ea6b6d88}}, // transcendental
    {{0x0000000000021d88, 0x00000001c8690400}}, // complex
    {{0x0000000000000000, 0x0000000100000000}}, // imaginary
    {{0x0000000000021d88, 0x0000000000000000}}, // hermitian
    {{0x0000000000000000, 0x0000000000000000}}, // antihermitian
    {{0x0000000000000000, 0x0000000000000000}}, // polar
}};
constexpr std::array<uint32_t, kNumFacts> kPrereq = {
    0x0ffffe7e, // commutative
    0x1ff003fd, // integer
    0x08f00253, // noninteger
    0x1ff003f3, // rational
    0x1cf003eb, // irrational
    0x1cf07fdb, // real
    0x08fffe3f, // extended_real
    0x0ff07f3a, // finite
    0x0ff07eba, // infinite
    0x1ffffdff, // zero
    0x1ce81bf1, // nonzero
    0x1ccff7e1, // positive
    0x1ccfefe1, // negative
    0x1cc31be1, // nonnegative
    0x1cc49be1, // nonpositive
    0x08cf5a41, // extended_positive
    0x08ceba41, // extended_negative
    0x08c1ba41, // extended_nonnegative
    0x08c1da41, // extended_nonpositive
    0x08e19e55, // extended_nonzero
    0x1f2003ff, // even
    0x1f1003ff, // odd
    0x1f8fffff, // prime
    0x1f4fffff, // composite
    0x06f0038b, // algebraic
    0x05f0038b, // transcendental
    0x0bf07fbb, // complex
    0x24ffffff, // imaginary
    0x00f07e3a, // hermitian
    0x08000000, // antihermitian
    0x00000000, // polar
};
// clang-format on
static_assert(kBetaRules.size() <= 64);
static_assert(kNumFacts <= 32);

size_t idx(Fact f) {
  return static_cast<size_t>(f);
}

uint32_t bit(Fact f) {
  return 1u << static_cast<unsigned>(f);
}

Tri tri(bool b) {
  return b ? Tri::True : Tri::False;
}

uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
  return static_cast<uint64_t>(static_cast<unsigned __int128>(a) * b % m);
}

uint64_t powmod(uint64_t b, uint64_t e, uint64_t m) {
  uint64_t r = 1;
  for (b %= m; e != 0; e >>= 1) {
    if (e & 1) {
      r = mulmod(r, b, m);
    }
    b = mulmod(b, b, m);
  }
  return r;
}

// sympy.isprime; Miller-Rabin with these bases is exact below 2**64.
bool is_prime(int64_t v) {
  if (v < 2) {
    return false;
  }
  auto n = static_cast<uint64_t>(v);
  static constexpr std::array<uint64_t, 12> bases = {
      2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
  for (uint64_t p : bases) {
    if (n % p == 0) {
      return n == p;
    }
  }
  uint64_t d = n - 1;
  int s = 0;
  for (; d % 2 == 0; d /= 2) {
    ++s;
  }
  for (uint64_t a : bases) {
    uint64_t x = powmod(a, d, n);
    if (x == 1 || x == n - 1) {
      continue;
    }
    int i = 1;
    for (; i < s; ++i) {
      x = mulmod(x, x, n);
      if (x == n - 1) {
        break;
      }
    }
    if (i == s) {
      return false;
    }
  }
  return true;
}

using F = Fact;

bool holds(ExprArena& A, const Expr* e, Fact f) {
  return A.ask(e, f) == Tri::True;
}

bool fails(ExprArena& A, const Expr* e, Fact f) {
  return A.ask(e, f) == Tri::False;
}

Tri fuzzy_not(Tri t) {
  return t == Tri::Unknown ? t : tri(t == Tri::False);
}

// torch.utils._sympy.functions._torf((a.is_<f> for a in args)).
Tri torf(ExprArena& A, c10::ArrayRef<const Expr*> args, Fact f) {
  bool saw_true = false;
  bool saw_false = false;
  for (const Expr* a : args) {
    Tri v = A.ask(a, f);
    if (v == Tri::Unknown || (v == Tri::True ? saw_false : saw_true)) {
      return Tri::Unknown;
    }
    (v == Tri::True ? saw_true : saw_false) = true;
  }
  return tri(saw_true);
}

// fuzzy_and((a.is_<f> for a in args)), or fuzzy_or when `any` is set.
Tri fuzzy_and_or(
    ExprArena& A,
    c10::ArrayRef<const Expr*> args,
    Fact f,
    bool any) {
  Tri r = tri(!any);
  for (const Expr* a : args) {
    Tri v = A.ask(a, f);
    if (v == tri(any)) {
      return v;
    }
    if (v == Tri::Unknown) {
      r = v;
    }
  }
  return r;
}

// _fuzzy_group((a.is_<f> for a in args), quick_exit).
Tri fuzzy_group(
    ExprArena& A,
    c10::ArrayRef<const Expr*> args,
    Fact f,
    bool quick_exit) {
  bool saw_other = false;
  for (const Expr* a : args) {
    Tri v = A.ask(a, f);
    if (v == Tri::True) {
      continue;
    }
    if (v == Tri::Unknown || (quick_exit && saw_other)) {
      return Tri::Unknown;
    }
    saw_other = true;
  }
  return tri(!saw_other);
}

bool all_hold(ExprArena& A, c10::ArrayRef<const Expr*> args, Fact f) {
  return std::all_of(
      args.begin(), args.end(), [&](const Expr* a) { return holds(A, a, f); });
}

bool any_holds(ExprArena& A, c10::ArrayRef<const Expr*> args, Fact f) {
  return std::any_of(
      args.begin(), args.end(), [&](const Expr* a) { return holds(A, a, f); });
}

c10::ArrayRef<const Expr*> make_args(const Expr* const& e) {
  if (e->kind == Kind::Mul) {
    return e->args;
  }
  return c10::ArrayRef<const Expr*>(&e, 1);
}

// abs(e) is S.One.
bool is_unit(const Expr* e) {
  return e->kind == Kind::Integer && (e->p == 1 || e->p == -1);
}

int64_t trailing(int64_t n) {
  auto u = static_cast<uint64_t>(n);
  return n == 0 ? 0 : std::countr_zero(n < 0 ? -u : u);
}

// Add(*[f.as_base_exp()[1] for f in factors if f.is_even]) - trailing(n),
// where exponents are Integers (Pow bases are never Rational).
int64_t even_exponents_minus_trailing(
    ExprArena& A,
    c10::ArrayRef<const Expr*> factors,
    int64_t n) {
  int64_t r = -trailing(n);
  for (const Expr* f : factors) {
    if (holds(A, f, F::even) &&
        __builtin_add_overflow(
            r, f->kind == Kind::Pow ? f->args[1]->p : 1, &r)) {
      throw NativeUnsupported("integer overflow");
    }
  }
  return r;
}

// sympy.simplify.radsimp.fraction(e) (exact=False).
std::pair<const Expr*, const Expr*> fraction(ExprArena& A, const Expr* e) {
  c10::SmallVector<const Expr*, 8> numer;
  c10::SmallVector<const Expr*, 8> denom;
  for (const Expr* term : make_args(e)) {
    if (term->kind == Kind::Pow) {
      const Expr* b = term->args[0];
      const Expr* x = term->args[1];
      if (holds(A, x, F::negative)) {
        denom.push_back(x->p == -1 ? b : A.pow(b, A.neg(x)));
      } else {
        numer.push_back(term);
      }
    } else if (term->kind == Kind::Rational) {
      if (term->p != 1) {
        numer.push_back(A.integer(term->p));
      }
      denom.push_back(A.integer(term->q));
    } else {
      numer.push_back(term);
    }
  }
  return {A.mul(numer), A.mul(denom)};
}

// Mul._eval_is_zero_infinite_helper: (seen_zero, seen_infinite).
std::pair<Tri, Tri> mul_zero_infinite(ExprArena& A, const Expr* e) {
  constexpr std::pair<Tri, Tri> unknown{Tri::Unknown, Tri::Unknown};
  Tri seen_zero = Tri::False;
  Tri seen_infinite = Tri::False;
  for (const Expr* a : e->args) {
    if (holds(A, a, F::zero)) {
      if (seen_infinite != Tri::False) {
        return unknown;
      }
      seen_zero = Tri::True;
    } else if (holds(A, a, F::infinite)) {
      if (seen_zero != Tri::False) {
        return unknown;
      }
      seen_infinite = Tri::True;
    } else {
      if (seen_zero == Tri::False && A.ask(a, F::zero) == Tri::Unknown) {
        if (seen_infinite != Tri::False) {
          return unknown;
        }
        seen_zero = Tri::Unknown;
      }
      if (seen_infinite == Tri::False &&
          A.ask(a, F::infinite) == Tri::Unknown) {
        if (seen_zero != Tri::False) {
          return unknown;
        }
        seen_infinite = Tri::Unknown;
      }
    }
  }
  return {seen_zero, seen_infinite};
}

Tri mul_is_zero(ExprArena& A, const Expr* e) {
  auto [seen_zero, seen_infinite] = mul_zero_infinite(A, e);
  if (seen_zero == Tri::False) {
    return Tri::False;
  }
  if (seen_zero == Tri::True && seen_infinite == Tri::False) {
    return Tri::True;
  }
  return Tri::Unknown;
}

// Mul._eval_is_rational and Mul._eval_is_algebraic.
Tri mul_rational_like(ExprArena& A, const Expr* e, Fact f) {
  Tri r = fuzzy_group(A, e->args, f, true);
  if (r == Tri::False &&
      !std::all_of(e->args.begin(), e->args.end(), [&](const Expr* a) {
        return fails(A, a, F::zero);
      })) {
    return Tri::Unknown;
  }
  return r;
}

Tri mul_is_integer(ExprArena& A, const Expr* e) {
  if (mul_rational_like(A, e, F::rational) == Tri::False) {
    return Tri::False;
  }
  c10::SmallVector<const Expr*, 4> numerators;
  c10::SmallVector<const Expr*, 4> denominators;
  bool unknown = false;
  for (const Expr* a : e->args) {
    if (holds(A, a, F::integer)) {
      if (!is_unit(a)) {
        numerators.push_back(a);
      }
    } else if (a->kind == Kind::Rational) {
      if (a->p != 1 && a->p != -1) {
        numerators.push_back(A.integer(a->p));
      }
      denominators.push_back(A.integer(a->q));
    } else if (a->kind == Kind::Pow) {
      const Expr* b = a->args[0];
      const Expr* x = a->args[1];
      if (!holds(A, b, F::integer) || !holds(A, x, F::integer)) {
        unknown = true;
      }
      if (!holds(A, x, F::negative)) {
        return Tri::Unknown;
      }
      denominators.push_back(A.pow(a, A.integer(-1)));
    } else {
      return Tri::Unknown;
    }
  }
  if (denominators.empty() && !unknown) {
    return Tri::True;
  }
  if (numerators.empty() && !denominators.empty() &&
      std::all_of(denominators.begin(), denominators.end(), [&](auto d) {
        return fuzzy_not(A.is_ge(A.integer(1), d)) == Tri::True;
      })) {
    return Tri::False;
  }
  if (unknown) {
    return Tri::Unknown;
  }
  if (all_hold(A, numerators, F::odd) && any_holds(A, denominators, F::even)) {
    return Tri::False;
  }
  if (any_holds(A, numerators, F::even) && denominators.size() == 1 &&
      denominators[0]->kind == Kind::Integer && denominators[0]->p == 2) {
    return Tri::True;
  }
  if (all_hold(A, numerators, F::even) && all_hold(A, denominators, F::odd)) {
    // (Mul(*denominators, evaluate=False) - 1).is_positive: every Integer
    // denominator is at least 2.
    for (const Expr* d : denominators) {
      if (d->kind != Kind::Integer) {
        throw NativeUnsupported("unevaluated Mul in Mul._eval_is_integer");
      }
    }
    return Tri::False;
  }
  if (denominators.size() == 1) {
    const Expr* d = denominators[0];
    if (d->kind == Kind::Integer && holds(A, d, F::even) &&
        even_exponents_minus_trailing(A, numerators, d->p) >= 0) {
      return Tri::True;
    }
  }
  if (numerators.size() == 1) {
    const Expr* n = numerators[0];
    if (n->kind == Kind::Integer && holds(A, n, F::even) &&
        even_exponents_minus_trailing(A, denominators, n->p) > 0) {
      return Tri::False;
    }
  }
  return Tri::Unknown;
}

// Mul._eval_real_imag.
Tri mul_real_imag(ExprArena& A, const Expr* e, bool real) {
  Tri zero = Tri::False;
  const Expr* t_not_re_im = nullptr;
  for (const Expr* t : e->args) {
    Tri complex = A.ask(t, F::complex);
    Tri complex_or_infinite =
        complex == Tri::True ? complex : A.ask(t, F::infinite);
    if (complex_or_infinite == Tri::False && fails(A, t, F::extended_real)) {
      return Tri::False;
    } else if (holds(A, t, F::imaginary)) {
      real = !real;
    } else if (holds(A, t, F::extended_real)) {
      if (zero != Tri::True) {
        Tri z = A.ask(t, F::zero);
        if (z != Tri::True && zero == Tri::False) {
          zero = z;
        } else if (z == Tri::True) {
          if (all_hold(A, e->args, F::finite)) {
            return Tri::True;
          }
          return Tri::Unknown;
        }
      }
    } else if (fails(A, t, F::extended_real) || fails(A, t, F::imaginary)) {
      if (t_not_re_im) {
        return Tri::Unknown;
      }
      t_not_re_im = t;
    } else {
      return Tri::Unknown;
    }
  }
  if (t_not_re_im) {
    if (fails(A, t_not_re_im, F::extended_real) && real) {
      return zero;
    }
    if (fails(A, t_not_re_im, F::imaginary) && !real) {
      return zero;
    }
  } else if (zero == Tri::False || real) {
    return tri(real);
  }
  return Tri::Unknown;
}

// Mul._eval_herm_antiherm.
Tri mul_herm_antiherm(ExprArena& A, const Expr* e, bool herm) {
  for (const Expr* t : e->args) {
    Tri h = A.ask(t, F::hermitian);
    if (h == Tri::Unknown || A.ask(t, F::antihermitian) == Tri::Unknown) {
      return Tri::Unknown;
    }
    if (h == Tri::True) {
      continue;
    }
    if (holds(A, t, F::antihermitian)) {
      herm = !herm;
    } else {
      return Tri::Unknown;
    }
  }
  if (herm) {
    return Tri::True;
  }
  return mul_is_zero(A, e);
}

// Mul._eval_pos_neg.
Tri mul_pos_neg(ExprArena& A, const Expr* e, int sign) {
  bool saw_non = false;
  bool saw_not = false;
  for (const Expr* t : e->args) {
    if (holds(A, t, F::extended_positive)) {
      continue;
    } else if (holds(A, t, F::extended_negative)) {
      sign = -sign;
    } else if (holds(A, t, F::zero)) {
      if (all_hold(A, e->args, F::finite)) {
        return Tri::False;
      }
      return Tri::Unknown;
    } else if (holds(A, t, F::extended_nonpositive)) {
      sign = -sign;
      saw_non = true;
    } else if (holds(A, t, F::extended_nonnegative)) {
      saw_non = true;
    } else if (fails(A, t, F::positive)) {
      sign = -sign;
      if (saw_not) {
        return Tri::Unknown;
      }
      saw_not = true;
    } else if (fails(A, t, F::negative)) {
      if (saw_not) {
        return Tri::Unknown;
      }
      saw_not = true;
    } else {
      return Tri::Unknown;
    }
  }
  if (sign == 1 && !saw_non && !saw_not) {
    return Tri::True;
  }
  return sign < 0 ? Tri::False : Tri::Unknown;
}

Tri mul_fact(ExprArena& A, const Expr* e, Fact f) {
  const auto& args = e->args;
  switch (f) {
    case F::complex: {
      Tri comp = fuzzy_group(A, args, F::complex, false);
      if (comp == Tri::False && any_holds(A, args, F::infinite)) {
        for (const Expr* a : args) {
          if (!fails(A, a, F::zero)) {
            return Tri::Unknown;
          }
        }
      }
      return comp;
    }
    case F::zero:
      return mul_is_zero(A, e);
    case F::infinite: {
      auto [seen_zero, seen_infinite] = mul_zero_infinite(A, e);
      if (seen_infinite == Tri::True && seen_zero == Tri::False) {
        return Tri::True;
      }
      return seen_infinite == Tri::False ? Tri::False : Tri::Unknown;
    }
    case F::rational:
    case F::algebraic:
      return mul_rational_like(A, e, f);
    case F::integer:
      return mul_is_integer(A, e);
    case F::polar:
      return tri(
          any_holds(A, args, F::polar) &&
          std::all_of(args.begin(), args.end(), [&](const Expr* a) {
            return holds(A, a, F::polar) || holds(A, a, F::positive);
          }));
    case F::extended_real:
      return mul_real_imag(A, e, true);
    case F::imaginary:
      for (const Expr* a : args) {
        if (!fails(A, a, F::zero) || !holds(A, a, F::finite)) {
          return Tri::Unknown;
        }
      }
      return mul_real_imag(A, e, false);
    case F::hermitian:
    case F::antihermitian:
      return mul_herm_antiherm(A, e, f == F::hermitian);
    case F::irrational:
      for (const Expr* t : args) {
        Tri a = A.ask(t, F::irrational);
        if (a == Tri::True) {
          for (const Expr* x : args) {
            if (x != t && !(holds(A, x, F::rational) && fails(A, x, F::zero))) {
              return Tri::Unknown;
            }
          }
          return Tri::True;
        }
        if (a == Tri::Unknown) {
          return Tri::Unknown;
        }
      }
      return all_hold(A, args, F::real) ? Tri::False : Tri::Unknown;
    case F::extended_positive:
      return mul_pos_neg(A, e, 1);
    case F::extended_negative:
      return mul_pos_neg(A, e, -1);
    case F::odd: {
      Tri is_integer = mul_is_integer(A, e);
      if (is_integer != Tri::True) {
        return is_integer;
      }
      auto [n, d] = fraction(A, e);
      if (d->kind == Kind::Integer && holds(A, d, F::even)) {
        if (even_exponents_minus_trailing(A, make_args(n), d->p) > 0) {
          return Tri::False;
        }
        return Tri::Unknown;
      }
      Tri r = Tri::True;
      const Expr* acc = nullptr;
      for (const Expr* t : args) {
        if (is_unit(t)) {
          continue;
        }
        if (holds(A, t, F::even)) {
          return Tri::False;
        }
        if (r != Tri::False && acc && holds(A, A.add({acc, t}), F::odd)) {
          r = Tri::False;
        } else if (r != Tri::False && A.ask(t, F::even) == Tri::Unknown) {
          r = Tri::Unknown;
        }
        acc = t;
      }
      return r;
    }
    case F::even: {
      auto [n, d] = fraction(A, e);
      if (n->kind == Kind::Integer && holds(A, n, F::even) &&
          even_exponents_minus_trailing(A, make_args(d), n->p) >= 0) {
        return Tri::False;
      }
      return Tri::Unknown;
    }
    case F::composite: {
      int count = 0;
      for (const Expr* a : args) {
        if (!(holds(A, a, F::integer) && holds(A, a, F::positive))) {
          return Tri::Unknown;
        }
        if (holds(A, A.sub(a, A.integer(1)), F::positive)) {
          ++count;
        }
      }
      return count > 1 ? Tri::True : Tri::Unknown;
    }
    default:
      return Tri::Unknown;
  }
}

Tri pow_is_extended_negative(ExprArena& A, const Expr* b, const Expr* x) {
  if (holds(A, b, F::extended_negative)) {
    if (holds(A, x, F::odd) && holds(A, b, F::finite)) {
      return Tri::True;
    }
    if (holds(A, x, F::even)) {
      return Tri::False;
    }
  } else if (holds(A, b, F::extended_positive)) {
    if (holds(A, x, F::extended_real)) {
      return Tri::False;
    }
  } else if (holds(A, b, F::zero)) {
    if (holds(A, x, F::extended_real)) {
      return Tri::False;
    }
  } else if (holds(A, b, F::extended_nonnegative)) {
    if (holds(A, x, F::extended_nonnegative)) {
      return Tri::False;
    }
  } else if (
      holds(A, b, F::extended_nonpositive) || holds(A, b, F::extended_real)) {
    if (holds(A, x, F::even)) {
      return Tri::False;
    }
  }
  return Tri::Unknown;
}

Tri pow_is_finite(ExprArena& A, const Expr* b, const Expr* x) {
  if (holds(A, x, F::negative)) {
    if (holds(A, b, F::zero)) {
      return Tri::False;
    }
    if (holds(A, b, F::infinite) || holds(A, b, F::nonzero)) {
      return Tri::True;
    }
  }
  Tri c1 = A.ask(b, F::finite);
  if (c1 == Tri::Unknown) {
    return c1;
  }
  Tri c2 = A.ask(x, F::finite);
  if (c2 == Tri::Unknown) {
    return c2;
  }
  if (c1 == Tri::True && c2 == Tri::True &&
      (holds(A, x, F::nonnegative) || fails(A, b, F::zero))) {
    return Tri::True;
  }
  return Tri::Unknown;
}

// ExprArena::pow only builds Pows whose base is not a number and whose
// exponent is an Integer other than 0 and 1, so the Exp1, ImaginaryUnit and
// Rational-base branches of the Pow handlers cannot be taken.
Tri pow_fact(ExprArena& A, const Expr* e, Fact f) {
  const Expr* b = e->args[0];
  const Expr* x = e->args[1];
  switch (f) {
    case F::even:
      if (holds(A, x, F::integer) && holds(A, x, F::positive)) {
        return A.ask(b, F::even);
      }
      return Tri::Unknown;
    case F::negative: {
      Tri ext_neg = pow_is_extended_negative(A, b, x);
      return ext_neg == Tri::True ? A.ask(e, F::finite) : ext_neg;
    }
    case F::extended_positive:
      if (b == x) {
        if (holds(A, b, F::extended_nonnegative)) {
          return Tri::True;
        }
      } else if (holds(A, b, F::positive)) {
        if (holds(A, x, F::real)) {
          return Tri::True;
        }
      } else if (holds(A, b, F::extended_negative)) {
        if (holds(A, x, F::even)) {
          return Tri::True;
        }
        if (holds(A, x, F::odd)) {
          return Tri::False;
        }
      } else if (holds(A, b, F::zero)) {
        if (holds(A, x, F::extended_real)) {
          return A.ask(x, F::zero);
        }
      } else if (holds(A, b, F::extended_nonpositive)) {
        if (holds(A, x, F::odd)) {
          return Tri::False;
        }
      } else if (holds(A, b, F::imaginary)) {
        if (holds(A, x, F::integer)) {
          return tri(x->p % 4 == 0);
        }
      }
      return Tri::Unknown;
    case F::extended_negative:
      return pow_is_extended_negative(A, b, x);
    case F::zero:
      if (holds(A, b, F::zero)) {
        if (holds(A, x, F::extended_positive)) {
          return Tri::True;
        }
        if (holds(A, x, F::extended_nonpositive)) {
          return Tri::False;
        }
      } else if (fails(A, b, F::zero)) {
        if (holds(A, b, F::finite) && holds(A, x, F::finite)) {
          return Tri::False;
        }
        if (holds(A, x, F::negative)) {
          return A.ask(b, F::infinite);
        }
        if (holds(A, x, F::nonnegative)) {
          return Tri::False;
        }
      } else if (holds(A, b, F::finite) && holds(A, x, F::negative)) {
        return Tri::False;
      }
      return Tri::Unknown;
    case F::integer:
      if (holds(A, b, F::rational) && fails(A, b, F::integer) &&
          holds(A, x, F::positive)) {
        return Tri::False;
      }
      if (holds(A, b, F::integer) && holds(A, x, F::integer) &&
          (holds(A, x, F::nonnegative) || holds(A, x, F::positive))) {
        return Tri::True;
      }
      if (holds(A, b, F::integer) && holds(A, x, F::negative) &&
          (holds(A, x, F::finite) || holds(A, x, F::integer)) &&
          fails(A, A.sub(b, A.integer(1)), F::zero) &&
          fails(A, A.add({b, A.integer(1)}), F::zero)) {
        return Tri::False;
      }
      if (holds(A, x, F::negative) && holds(A, b, F::positive) &&
          holds(A, A.sub(b, A.integer(1)), F::positive)) {
        return Tri::False;
      }
      if (holds(A, x, F::negative) && holds(A, b, F::negative) &&
          holds(A, A.add({b, A.integer(1)}), F::negative)) {
        return Tri::False;
      }
      return Tri::Unknown;
    case F::extended_real: {
      Tri real_b = A.ask(b, F::extended_real);
      if (real_b == Tri::Unknown) {
        return real_b;
      }
      Tri real_e = A.ask(x, F::extended_real);
      if (real_e == Tri::Unknown) {
        return real_e;
      }
      if (real_b == Tri::True && real_e == Tri::True) {
        if (holds(A, b, F::extended_positive) ||
            (holds(A, b, F::extended_nonnegative) &&
             holds(A, x, F::extended_nonnegative)) ||
            (holds(A, x, F::integer) && holds(A, b, F::extended_nonzero)) ||
            (holds(A, x, F::integer) && holds(A, x, F::nonnegative))) {
          return Tri::True;
        }
        if (holds(A, b, F::extended_negative)) {
          return Tri::False;
        }
      }
      if (real_e == Tri::True && holds(A, x, F::extended_negative) &&
          fails(A, b, F::zero)) {
        return A.ask(A.pow(b, A.neg(x)), F::extended_real);
      }
      Tri im_b = A.ask(b, F::imaginary);
      if (im_b == Tri::True && holds(A, x, F::integer)) {
        if (holds(A, x, F::even)) {
          return Tri::True;
        }
        if (holds(A, x, F::odd)) {
          return Tri::False;
        }
      }
      if (real_b == Tri::False && real_e == Tri::True) {
        throw NativeUnsupported("Pow._eval_is_extended_real needs arg()");
      }
      return Tri::Unknown;
    }
    case F::complex:
      if (all_hold(A, e->args, F::complex) &&
          pow_is_finite(A, b, x) == Tri::True) {
        return Tri::True;
      }
      return Tri::Unknown;
    case F::imaginary:
      if (holds(A, b, F::imaginary) && holds(A, x, F::integer)) {
        return A.ask(x, F::odd);
      }
      if (holds(A, b, F::extended_real) && holds(A, x, F::extended_real)) {
        if (holds(A, b, F::positive)) {
          return Tri::False;
        }
        Tri rat = A.ask(x, F::rational);
        if (rat != Tri::True) {
          return rat;
        }
        if (holds(A, x, F::integer)) {
          return Tri::False;
        }
        throw NativeUnsupported("Pow with a non-integer exponent");
      }
      if (fails(A, b, F::extended_real)) {
        throw NativeUnsupported("Pow._eval_is_imaginary needs arg()");
      }
      return Tri::Unknown;
    case F::odd:
      if (holds(A, x, F::integer)) {
        if (holds(A, x, F::positive)) {
          return A.ask(b, F::odd);
        }
        if (holds(A, x, F::nonnegative) && holds(A, b, F::odd)) {
          return Tri::True;
        }
      }
      return Tri::Unknown;
    case F::finite:
      return pow_is_finite(A, b, x);
    case F::prime:
      if (holds(A, b, F::integer) && holds(A, x, F::integer) &&
          holds(A, A.sub(x, A.integer(1)), F::positive)) {
        return Tri::False;
      }
      return Tri::Unknown;
    case F::composite:
      if (holds(A, b, F::integer) && holds(A, x, F::integer) &&
          ((holds(A, A.sub(b, A.integer(1)), F::positive) &&
            holds(A, A.sub(x, A.integer(1)), F::positive)) ||
           (holds(A, A.add({b, A.integer(1)}), F::negative) &&
            holds(A, x, F::positive) && holds(A, x, F::even)))) {
        return Tri::True;
      }
      return Tri::Unknown;
    case F::polar:
      return A.ask(b, F::polar);
    case F::rational: {
      if (holds(A, x, F::integer) && holds(A, b, F::rational)) {
        Tri neg = A.ask(x, F::negative);
        if (neg == Tri::False || fails(A, b, F::zero)) {
          return Tri::True;
        }
      }
      if (holds(A, x, F::integer)) {
        if (holds(A, b, F::rational)) {
          if (fails(A, b, F::zero) || holds(A, x, F::nonnegative)) {
            return Tri::True;
          }
        } else if (holds(A, b, F::irrational)) {
          return A.ask(x, F::zero);
        }
      }
      return Tri::Unknown;
    }
    case F::algebraic:
      if (holds(A, b, F::zero) || holds(A, A.sub(b, A.integer(1)), F::zero)) {
        return Tri::True;
      }
      if (holds(A, x, F::rational)) {
        if (fails(A, b, F::algebraic)) {
          return A.ask(x, F::zero);
        }
        if (fails(A, b, F::zero)) {
          if (holds(A, x, F::nonzero)) {
            return A.ask(b, F::algebraic);
          }
          if (holds(A, b, F::algebraic)) {
            return Tri::True;
          }
        }
        if (holds(A, x, F::positive)) {
          return A.ask(b, F::algebraic);
        }
      }
      return Tri::Unknown;
    default:
      return Tri::Unknown;
  }
}

// Add._eval_is_extended_positive, or _eval_is_extended_negative with the signs
// mirrored.
Tri add_pos_neg(ExprArena& A, const Expr* e, bool positive) {
  const Fact pos = positive ? F::extended_positive : F::extended_negative;
  const Fact nonneg =
      positive ? F::extended_nonnegative : F::extended_nonpositive;
  const Fact nonpos =
      positive ? F::extended_nonpositive : F::extended_nonnegative;
  auto [c, a] = A.as_coeff_Add(e);
  if (c->p != 0) {
    const Expr* v = A.monotonic_sign(a);
    if (v != nullptr) {
      const Expr* s = A.add({v, c});
      if (s != e && holds(A, s, pos) && holds(A, a, nonneg)) {
        return Tri::True;
      }
      if (A.free_symbols(e).size() == 1) {
        v = A.monotonic_sign(e);
        if (v != nullptr && v != e && holds(A, v, pos)) {
          return Tri::True;
        }
      }
    }
  }
  bool saw_pos = false;
  bool saw_nonneg = false;
  bool saw_nonpos = false;
  bool unknown_sign = false;
  // The set saw_INF as a bitmask over Tri values.
  unsigned saw_inf = 0;
  c10::SmallVector<const Expr*, 8> args;
  for (const Expr* t : e->args) {
    if (!holds(A, t, F::zero)) {
      args.push_back(t);
    }
  }
  if (args.empty()) {
    return Tri::False;
  }
  for (const Expr* t : args) {
    Tri ispos = A.ask(t, pos);
    Tri infinite = A.ask(t, F::infinite);
    if (infinite == Tri::True) {
      Tri isnonneg = A.ask(t, nonneg);
      Tri any = ispos == Tri::True || isnonneg == Tri::True   ? Tri::True
          : ispos == Tri::Unknown || isnonneg == Tri::Unknown ? Tri::Unknown
                                                              : Tri::False;
      saw_inf |= 1u << static_cast<unsigned>(any);
      if ((saw_inf & 3) == 3) {
        return Tri::Unknown;
      }
    }
    if (ispos == Tri::True) {
      saw_pos = true;
      continue;
    } else if (holds(A, t, nonneg)) {
      saw_nonneg = true;
      continue;
    } else if (holds(A, t, nonpos)) {
      saw_nonpos = true;
      continue;
    }
    if (infinite == Tri::Unknown) {
      return Tri::Unknown;
    }
    unknown_sign = true;
  }
  if (saw_inf != 0) {
    if (std::popcount(saw_inf) > 1) {
      return Tri::Unknown;
    }
    return static_cast<Tri>(std::countr_zero(saw_inf));
  } else if (unknown_sign) {
    return Tri::Unknown;
  } else if (!saw_nonpos && saw_pos) {
    return Tri::True;
  } else if (!saw_pos && !saw_nonneg) {
    return Tri::False;
  }
  return Tri::Unknown;
}

// Add._eval_is_extended_nonnegative, or _eval_is_extended_nonpositive.
Tri add_nonneg_nonpos(ExprArena& A, const Expr* e, Fact f) {
  auto [c, a] = A.as_coeff_Add(e);
  if (c->p != 0 && holds(A, a, f)) {
    const Expr* v = A.monotonic_sign(a);
    if (v != nullptr) {
      const Expr* s = A.add({v, c});
      if (s != e && holds(A, s, f)) {
        return Tri::True;
      }
      if (A.free_symbols(e).size() == 1) {
        v = A.monotonic_sign(e);
        if (v != nullptr && v != e && holds(A, v, f)) {
          return Tri::True;
        }
      }
    }
  }
  return Tri::Unknown;
}

Tri add_fact(ExprArena& A, const Expr* e, Fact f) {
  const auto& args = e->args;
  switch (f) {
    case F::real:
    case F::extended_real:
    case F::complex:
    case F::antihermitian:
    case F::finite:
    case F::hermitian:
    case F::integer:
    case F::rational:
    case F::algebraic:
      return fuzzy_group(A, args, f, true);
    case F::infinite: {
      bool sawinf = false;
      for (const Expr* a : args) {
        Tri ainf = A.ask(a, F::infinite);
        if (ainf == Tri::Unknown || (ainf == Tri::True && sawinf)) {
          return Tri::Unknown;
        }
        sawinf |= ainf == Tri::True;
      }
      return tri(sawinf);
    }
    case F::imaginary: {
      c10::SmallVector<const Expr*, 8> nz;
      for (const Expr* a : args) {
        if (holds(A, a, F::extended_real)) {
          Tri z = A.ask(a, F::zero);
          if (z == Tri::False) {
            nz.push_back(a);
          } else if (z == Tri::Unknown) {
            return Tri::Unknown;
          }
        } else if (holds(A, a, F::imaginary)) {
          throw NativeUnsupported("Add._eval_is_imaginary needs I");
        } else {
          return Tri::Unknown;
        }
      }
      const Expr* b = A.add(nz);
      if (b != e) {
        // Every term is real, so Add(*im_I) is 0.
        return A.ask(b, F::zero) == Tri::Unknown ? Tri::Unknown : Tri::False;
      }
      return Tri::Unknown;
    }
    case F::zero: {
      c10::SmallVector<const Expr*, 8> nz;
      size_t z = 0;
      size_t im = 0;
      for (const Expr* a : args) {
        if (holds(A, a, F::extended_real)) {
          Tri az = A.ask(a, F::zero);
          if (az == Tri::True) {
            ++z;
          } else if (az == Tri::False) {
            nz.push_back(a);
          } else {
            return Tri::Unknown;
          }
        } else if (holds(A, a, F::imaginary)) {
          ++im;
        } else {
          return Tri::Unknown;
        }
      }
      if (z == args.size()) {
        return Tri::True;
      }
      if (nz.empty() || nz.size() == args.size()) {
        return Tri::Unknown;
      }
      Tri bz = A.ask(A.add(nz), F::zero);
      if (bz == Tri::True && im <= 1) {
        return tri(im == 0);
      }
      return bz == Tri::False ? Tri::False : Tri::Unknown;
    }
    case F::odd: {
      c10::SmallVector<const Expr*, 8> l;
      for (const Expr* a : args) {
        if (!holds(A, a, F::even)) {
          l.push_back(a);
        }
      }
      if (l.empty()) {
        return Tri::False;
      }
      if (holds(A, l[0], F::odd)) {
        // _new_rawargs(*l[1:]): a subset of canonical args is canonical.
        return A.ask(A.add(c10::ArrayRef<const Expr*>(l).slice(1)), F::even);
      }
      return Tri::Unknown;
    }
    case F::irrational:
      for (const Expr* t : args) {
        Tri a = A.ask(t, F::irrational);
        if (a == Tri::True) {
          for (const Expr* x : args) {
            if (x != t && !holds(A, x, F::rational)) {
              return Tri::Unknown;
            }
          }
          return Tri::True;
        }
        if (a == Tri::Unknown) {
          return Tri::Unknown;
        }
      }
      return Tri::False;
    case F::extended_positive:
    case F::extended_negative:
      return add_pos_neg(A, e, f == F::extended_positive);
    case F::extended_nonnegative:
    case F::extended_nonpositive:
      return add_nonneg_nonpos(A, e, f);
    default:
      return Tri::Unknown;
  }
}

} // namespace

AssumeRules assume_rules() {
  return {kFullImplications, kBetaRules, kBetaTriggers, kPrereq};
}

void deduce_all_facts(FactKB& kb, c10::ArrayRef<std::pair<Fact, bool>> in) {
  c10::SmallVector<std::pair<Fact, bool>, 8> facts(in.begin(), in.end());
  while (!facts.empty()) {
    uint64_t maytrigger = 0;
    for (auto [k, v] : facts) {
      // _tell(k, v) and then _tell of each implication, checking for
      // InconsistentAssumptions before changing the KB.
      uint32_t b = bit(k);
      if ((v ? kb.true_mask : kb.false_mask) & b) {
        continue;
      }
      const Implication& imp = kFullImplications[idx(k)][v];
      uint32_t t = imp.true_mask | (v ? b : 0);
      uint32_t f = imp.false_mask | (v ? 0 : b);
      if ((t & kb.false_mask) || (f & kb.true_mask)) {
        throw NativeUnsupported(
            std::string("inconsistent assumptions from ") + fact_name(k));
      }
      kb.true_mask |= t;
      kb.false_mask |= f;
      kb.known |= t | f;
      maytrigger |= kBetaTriggers[idx(k)][v];
    }
    facts.clear();
    for (; maytrigger != 0; maytrigger &= maytrigger - 1) {
      const BetaRule& r = kBetaRules[std::countr_zero(maytrigger)];
      if ((kb.true_mask & r.cond_true) == r.cond_true &&
          (kb.false_mask & r.cond_false) == r.cond_false) {
        facts.emplace_back(r.fact, r.value);
      }
    }
  }
}

FactKB ExprArena::default_kb(const Expr* e) {
  // StdFactKB(cls._explicit_class_assumptions) of each number and function
  // class.
  static const auto kbs = [] {
    using F = Fact;
    auto make = [](std::initializer_list<std::pair<Fact, bool>> extra) {
      FactKB kb;
      deduce_all_facts(kb, extra);
      return kb;
    };
    // Float's rational and irrational are None.
    FactKB float_kb = make(
        {{F::commutative, true}, {F::real, true}, {F::extended_real, true}});
    float_kb.known |= bit(F::rational) | bit(F::irrational);
    return std::array<FactKB, 13>{
        // Integer, NegativeOne
        make(
            {{F::commutative, true},
             {F::integer, true},
             {F::real, true},
             {F::rational, true}}),
        // Zero
        make(
            {{F::commutative, true},
             {F::integer, true},
             {F::real, true},
             {F::rational, true},
             {F::zero, true},
             {F::negative, false},
             {F::positive, false}}),
        // One
        make(
            {{F::commutative, true},
             {F::integer, true},
             {F::real, true},
             {F::rational, true},
             {F::positive, true}}),
        // Rational, Half
        make(
            {{F::commutative, true},
             {F::integer, false},
             {F::real, true},
             {F::rational, true}}),
        // IntInfinity
        make(
            {{F::commutative, true},
             {F::integer, true},
             {F::extended_positive, true},
             {F::extended_real, true},
             {F::prime, false}}),
        // NegativeIntInfinity
        make(
            {{F::commutative, true},
             {F::integer, true},
             {F::extended_negative, true},
             {F::extended_real, true},
             {F::prime, false}}),
        // is_integer = True
        make({{F::integer, true}}),
        // Mod
        make({{F::integer, true}, {F::nonnegative, true}}),
        // LatticeOp.is_commutative
        make({{F::commutative, true}}),
        // is_real = True
        make({{F::real, true}}),
        float_kb,
        // Infinity
        make(
            {{F::commutative, true},
             {F::complex, false},
             {F::extended_real, true},
             {F::infinite, true},
             {F::extended_positive, true},
             {F::prime, false}}),
        // NegativeInfinity
        make(
            {{F::commutative, true},
             {F::complex, false},
             {F::extended_real, true},
             {F::infinite, true},
             {F::extended_negative, true},
             {F::prime, false}}),
    };
  }();
  switch (e->kind) {
    case Kind::Integer:
      return kbs[e->p == 0 ? 1 : e->p == 1 ? 2 : 0];
    case Kind::Rational:
      return kbs[3];
    case Kind::Float:
      return kbs[10];
    case Kind::IntInfinity:
      return kbs[4];
    case Kind::NegativeIntInfinity:
      return kbs[5];
    case Kind::Infinity:
      return kbs[11];
    case Kind::NegativeInfinity:
      return kbs[12];
    case Kind::PythonMod:
    case Kind::FloorDiv:
    case Kind::CleanDiv:
    case Kind::PowByNatural:
    case Kind::CeilToInt:
    case Kind::FloorToInt:
    case Kind::TruncToInt:
    case Kind::RoundToInt:
    case Kind::IsNonOverlappingAndDenseIndicator:
    case Kind::ModularIndexing:
      return kbs[6];
    case Kind::Mod:
      return kbs[7];
    case Kind::Max:
    case Kind::Min:
      return kbs[8];
    case Kind::FloatPow:
    case Kind::FloatTrueDiv:
    case Kind::IntTrueDiv:
    case Kind::RoundDecimal:
    case Kind::ToFloat:
    case Kind::TruncToFloat:
      return kbs[9];
    default:
      return {};
  }
}

Tri ExprArena::ask(const Expr* e, Fact fact) {
  // _ask (sympy/core/assumptions.py): breadth-first over the facts that can
  // determine `fact`, in Fact order where sympy shuffles (the result does not
  // depend on the order when handlers agree with the rules).
  if (e->kind == Kind::Float &&
      (fact == F::rational || fact == F::irrational)) {
    // Float's class attributes shadow what its KB deduces.
    return Tri::Unknown;
  }
  FactKB& kb = e->kb;
  uint32_t queued = bit(fact);
  if (kb.known & queued) {
    return kb.get(fact);
  }
  c10::SmallVector<Fact, kNumFacts> queue{fact};
  for (size_t i = 0; i < queue.size(); ++i) {
    Fact fi = queue[i];
    if (kb.known & bit(fi)) {
      continue;
    }
    Tri v = eval_fact(e, fi);
    if (v != Tri::Unknown) {
      std::pair<Fact, bool> fv{fi, v == Tri::True};
      deduce_all_facts(kb, fv);
    }
    Tri r = kb.get(fact);
    if (r != Tri::Unknown) {
      return r;
    }
    uint32_t more = kPrereq[idx(fi)] & ~queued;
    queued |= more;
    for (; more != 0; more &= more - 1) {
      queue.push_back(static_cast<Fact>(std::countr_zero(more)));
    }
  }
  kb.known |= bit(fact);
  return kb.get(fact);
}

Tri ExprArena::eval_fact(const Expr* e, Fact f) {
  // The _eval_is_<f> handler of e's class, or Unknown if it has none.
  switch (e->kind) {
    case Kind::Integer:
      if (f == Fact::composite) {
        if (e->p <= 1) {
          return Tri::False;
        }
        Tri prime = ask(e, Fact::prime);
        return prime == Tri::Unknown ? prime : tri(prime == Tri::False);
      }
      if (f == Fact::odd) {
        return tri(e->p % 2 != 0);
      }
      if (f == Fact::prime) {
        return tri(is_prime(e->p));
      }
      [[fallthrough]];
    case Kind::Rational:
      switch (f) {
        case Fact::extended_negative:
          return tri(e->p < 0);
        case Fact::extended_positive:
        case Fact::positive:
          return tri(e->p > 0);
        case Fact::zero:
          return tri(e->p == 0);
        default:
          return Tri::Unknown;
      }
    case Kind::Float: {
      double v = e->float_value();
      switch (f) {
        case Fact::extended_negative:
        case Fact::negative:
          return tri(v < 0);
        case Fact::extended_positive:
        case Fact::positive:
          return tri(v > 0);
        case Fact::zero:
          return tri(v == 0);
        case Fact::integer:
          // None when int_valued.
          return v == 0            ? Tri::True
              : std::trunc(v) != v ? Tri::False
                                   : Tri::Unknown;
        default:
          return Tri::Unknown;
      }
    }
    case Kind::IntInfinity:
    case Kind::NegativeIntInfinity:
    case Kind::Infinity:
    case Kind::NegativeInfinity:
    case Kind::Symbol:
      // Their only handlers decide extended_positive/negative, which their KBs
      // already know or which are None for symbols.
      return Tri::Unknown;
    case Kind::Pow:
    case Kind::Mul:
    case Kind::Add:
      if (f == Fact::commutative) {
        // AssocOp and Pow keep is_commutative in a slot, not in the KB.
        return fuzzy_group(*this, e->args, f, false);
      }
      return e->kind == Kind::Pow ? pow_fact(*this, e, f)
          : e->kind == Kind::Mul  ? mul_fact(*this, e, f)
                                  : add_fact(*this, e, f);
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
      // Expr's extended_positive/negative handlers return None unless
      // is_number, and function nodes always have free symbols.
      // Function._eval_is_commutative is shadowed by the class facts.
      if (e->kind == Kind::PythonMod &&
          (f == F::nonnegative || f == F::nonpositive)) {
        Fact sign = f == F::nonnegative ? F::positive : F::negative;
        return ask(e->args[1], sign) == Tri::True ? Tri::True : Tri::Unknown;
      }
      return Tri::Unknown;
    case Kind::Where:
      if (f == F::integer || f == F::nonnegative || f == F::positive) {
        return ask(e->args[1], f) == Tri::True &&
                ask(e->args[2], f) == Tri::True
            ? Tri::True
            : Tri::Unknown;
      }
      [[fallthrough]];
    case Kind::BitwiseAnd:
    case Kind::BitwiseOr:
    case Kind::BitwiseXor:
    case Kind::OpaqueSqrt:
    case Kind::OpaqueCos:
    case Kind::OpaqueCosh:
    case Kind::OpaqueSin:
    case Kind::OpaqueSinh:
    case Kind::OpaqueTan:
    case Kind::OpaqueTanh:
    case Kind::OpaqueAsin:
    case Kind::OpaqueAcos:
    case Kind::OpaqueAtan:
    case Kind::OpaqueExp:
    case Kind::OpaqueLog:
    case Kind::OpaqueAsinh:
    case Kind::OpaqueLog2:
      // Function._eval_is_commutative; BitwiseFn and OpaqueUnaryFn have no
      // other facts.
      return f == F::commutative ? fuzzy_and_or(*this, e->args, f, false)
                                 : Tri::Unknown;
    case Kind::Identity:
      // _eval_is_real, _eval_is_integer, Function._eval_is_commutative.
      return f == F::real || f == F::integer || f == F::commutative
          ? ask(e->args[0], f)
          : Tri::Unknown;
    case Kind::ModularIndexing:
      if (f == F::nonnegative) {
        // fuzzy_eq(p.is_nonnegative, q.is_nonnegative)
        Tri p = ask(e->args[0], F::nonnegative);
        Tri q = ask(e->args[1], F::nonnegative);
        return p == Tri::Unknown || q == Tri::Unknown ? Tri::Unknown
                                                      : tri(p == q);
      }
      return Tri::Unknown;
    case Kind::FloorDiv:
    case Kind::CleanDiv:
      if (f == F::nonnegative) {
        // all([p.is_integer, q.is_integer, p.is_nonnegative, q.is_nonnegative])
        Tri facts[] = {
            ask(e->args[0], F::integer),
            ask(e->args[1], F::integer),
            ask(e->args[0], F::nonnegative),
            ask(e->args[1], F::nonnegative)};
        return std::all_of(
                   std::begin(facts),
                   std::end(facts),
                   [](Tri t) { return t == Tri::True; })
            ? Tri::True
            : Tri::Unknown;
      }
      return Tri::Unknown;
    case Kind::Max:
    case Kind::Min: {
      // MinMaxBase's _torf handlers for every fact but the extended signs,
      // with Max/Min overriding positive, nonnegative and negative.
      // is_commutative is a class fact.
      bool is_max = e->kind == Kind::Max;
      switch (f) {
        case F::positive:
        case F::nonnegative:
          return fuzzy_and_or(*this, e->args, f, is_max);
        case F::negative:
          return fuzzy_and_or(*this, e->args, f, !is_max);
        case F::extended_positive:
        case F::extended_negative:
        case F::extended_nonnegative:
        case F::extended_nonpositive:
        case F::extended_nonzero:
          return Tri::Unknown;
        default:
          return torf(*this, e->args, f);
      }
    }
    case Kind::BooleanTrue:
    case Kind::BooleanFalse:
    case Kind::Eq:
    case Kind::Ne:
    case Kind::Lt:
    case Kind::Le:
    case Kind::Gt:
    case Kind::Ge:
    case Kind::Not:
      // sympy Booleans have no assumption handlers or class facts.
      return Tri::Unknown;
    case Kind::And:
    case Kind::Or:
      // LatticeOp.is_commutative.
      return f == F::commutative ? Tri::True : Tri::Unknown;
  }
  return Tri::Unknown;
}

} // namespace torch::symbolic
