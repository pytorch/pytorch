#include <torch/csrc/symbolic/Expr.h>

#include <bit>
#include <initializer_list>

namespace torch::symbolic {

namespace {

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
  // StdFactKB(cls._explicit_class_assumptions) of each number class.
  static const auto kbs = [] {
    using F = Fact;
    auto make = [](std::initializer_list<std::pair<Fact, bool>> extra) {
      FactKB kb;
      deduce_all_facts(kb, extra);
      return kb;
    };
    return std::array<FactKB, 6>{
        // Integer, NegativeOne
        make({{F::commutative, true},
              {F::integer, true},
              {F::real, true},
              {F::rational, true}}),
        // Zero
        make({{F::commutative, true},
              {F::integer, true},
              {F::real, true},
              {F::rational, true},
              {F::zero, true},
              {F::negative, false},
              {F::positive, false}}),
        // One
        make({{F::commutative, true},
              {F::integer, true},
              {F::real, true},
              {F::rational, true},
              {F::positive, true}}),
        // Rational, Half
        make({{F::commutative, true},
              {F::integer, false},
              {F::real, true},
              {F::rational, true}}),
        // IntInfinity
        make({{F::commutative, true},
              {F::integer, true},
              {F::extended_positive, true},
              {F::extended_real, true},
              {F::prime, false}}),
        // NegativeIntInfinity
        make({{F::commutative, true},
              {F::integer, true},
              {F::extended_negative, true},
              {F::extended_real, true},
              {F::prime, false}}),
    };
  }();
  switch (e->kind) {
    case Kind::Integer:
      return kbs[e->p == 0 ? 1 : e->p == 1 ? 2 : 0];
    case Kind::Rational:
      return kbs[3];
    case Kind::IntInfinity:
      return kbs[4];
    case Kind::NegativeIntInfinity:
      return kbs[5];
    default:
      return {};
  }
}

Tri ExprArena::ask(const Expr* e, Fact fact) {
  // _ask (sympy/core/assumptions.py): breadth-first over the facts that can
  // determine `fact`, in Fact order where sympy shuffles (the result does not
  // depend on the order when handlers agree with the rules).
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
    case Kind::IntInfinity:
    case Kind::NegativeIntInfinity:
    case Kind::Symbol:
      // Their only handlers decide extended_positive/negative, which their KBs
      // already know or which are None for symbols.
      return Tri::Unknown;
    default:
      throw NativeUnsupported("assumptions of Add/Mul/Pow are not ported yet");
  }
}

} // namespace torch::symbolic
