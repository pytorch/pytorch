#pragma once

#include <charconv>
#include <cstdint>
#include <string>
#include <system_error>
#include <vector>

// Leading-dimension rules for GEMM signatures, split out of GemmCommon.h so
// they can be unit tested without pulling in ATen or CUDA. Pure integer,
// character and string logic -- no ATen or CUDA dependencies.
//
// Two sites share the transpose/tightness rules and must not drift apart:
//   * ShouldWildcardLd* in GemmCommon.h decides whether an ld may be replaced
//     by '*' when a logical dim is dynamic. It checks a SINGLE shape inline
//     (ld == its logical extent) because there is no second shape to compare.
//   * LdMayVary + LdsCompatibleWhen*Varies below compare TWO shapes when only
//     one logical dim differs, requiring tightness in both.
// What is shared is UsesMForLda/UsesKForLdb (which ld tracks which dim) and
// the tightness principle (a padded ld is not a function of m/n/k); the call
// shapes differ because the inputs differ.
//
// This header also hosts the signature parser (ParseGemmShape) and the
// borrow-compatibility check (NearestLargerMatch) used by
// TuningResultsManager::LookupNearest, so the grammar lives next to the rules
// and both are covered by the same unit test.

namespace at::cuda::tunable {

inline bool UsesMForLda(char transa) {
  // lda is the leading dim of A. For non-transposed A (transa == 'N'), A is
  // laid out with lda >= m, so lda tracks M; for any transposed form
  // ('T'/'t' and the conjugate 'C'/'c'), lda >= k, so lda tracks K. This
  // mirrors GetSizeA's stride computation.
  return transa == 'N' || transa == 'n';
}

inline bool UsesKForLdb(char transb) {
  return transb == 'N' || transb == 'n';
}

// Leading dims of one GEMM signature, for comparing two shapes that differ in a
// single logical dim.
struct GemmLeadingDims {
  int64_t lda = 0;
  int64_t ldb = 0;
  int64_t ldc = 0;
};

// A leading dim that tracks the varying logical dim may differ between the two
// shapes only when it is tight in BOTH. Same tightness rule as
// ShouldWildcardLd*: a padded ld is not a function of m/n/k, so treating it as
// one would pair layouts with different padding/alignment.
inline bool LdMayVary(int64_t q_ld, int64_t q_dim, int64_t c_ld, int64_t c_dim) {
  return q_ld == c_ld || (q_ld == q_dim && c_ld == c_dim);
}

// Are q's and c's leading dims compatible when only M differs? ldc always
// tracks M; lda tracks M only for non-transposed A (for 'T'/'t' and 'C'/'c'
// lda tracks K instead and must match exactly); ldb never depends on M.
inline bool LdsCompatibleWhenMVaries(
    char transa,
    const GemmLeadingDims& q, int64_t q_m,
    const GemmLeadingDims& c, int64_t c_m) {
  if (q.ldb != c.ldb) {
    return false;
  }
  if (UsesMForLda(transa)) {
    if (!LdMayVary(q.lda, q_m, c.lda, c_m)) {
      return false;
    }
  } else if (q.lda != c.lda) {
    return false;
  }
  return LdMayVary(q.ldc, q_m, c.ldc, c_m);
}

// Are q's and c's leading dims compatible when only N differs? Only ldb can
// depend on N, and only when B is transposed (UsesKForLdb false); lda and ldc
// never do.
inline bool LdsCompatibleWhenNVaries(
    char transb,
    const GemmLeadingDims& q, int64_t q_n,
    const GemmLeadingDims& c, int64_t c_n) {
  if (q.lda != c.lda || q.ldc != c.ldc) {
    return false;
  }
  if (UsesKForLdb(transb)) {
    return q.ldb == c.ldb;
  }
  return LdMayVary(q.ldb, q_n, c.ldb, c_n);
}

// --- Signature parsing and borrow matching for LookupNearest ---
//
// Grammar produced by the Signature() builders in GemmCommon.h:
//   ab_M_N_K_[B_<b>_]ld_lda_ldb_ldc[_<suffix...>]
// e.g. "NN_1024_512_256_ld_1024_256_1024",
//      "NN_8_512_256_B_4_ld_512_256_512",
//      "NN_1024_512_256_ld_1024_256_1024_rw_1_bias_BFloat16".
// Wildcard '*' tokens fail numeric conversion, so wildcard (dynamic) signatures
// never parse: borrowing is concrete-to-concrete by design.

inline std::vector<std::string> SplitUnderscore(const std::string& s) {
  std::vector<std::string> out;
  size_t start = 0;
  while (true) {
    const size_t pos = s.find('_', start);
    if (pos == std::string::npos) {
      out.push_back(s.substr(start));
      break;
    }
    out.push_back(s.substr(start, pos - start));
    start = pos + 1;
  }
  return out;
}

// Parsed form of a params signature: ab_M_N_K_[B_<b>_]ld_lda_ldb_ldc[_<suffix...>].
struct GemmShape {
  bool ok = false;
  std::string transab;  // [0]=transa, [1]=transb
  int64_t m = 0, n = 0, k = 0;
  bool has_b = false;
  int64_t b = 0;
  int64_t lda = 0, ldb = 0, ldc = 0;
  std::string suffix;  // tail after ldc (e.g. scaled_mm "_rw_1_bias_BFloat16")
};

// Parse a token that must be a whole non-negative decimal integer. Unlike
// std::stoll this neither throws nor accepts a prefix match: "12abc", " 12",
// "+12" and "" are all rejected, so a malformed signature cannot parse as ok.
// It also matters for cost: every wildcard ('*') entry in the results map is
// offered to this parser on every LookupNearest call, and throwing an
// exception per candidate to reject them is far more expensive than a failed
// from_chars. A leading '-' is rejected too -- no dim or leading dim is
// negative, so that is a malformed signature rather than a value to accept.
inline bool ParseWholeInt(const std::string& tok, int64_t& out) {
  if (tok.empty() || tok.front() == '-') {
    return false;
  }
  const char* const first = tok.data();
  const char* const last = first + tok.size();
  const auto [ptr, ec] = std::from_chars(first, last, out);
  return ec == std::errc{} && ptr == last;
}

inline GemmShape ParseGemmShape(const std::vector<std::string>& t) {
  GemmShape s;
  if (t.size() < 8 || t[0].size() != 2) {
    return s;
  }
  size_t i = 0;
  s.transab = t[i++];
  if (!ParseWholeInt(t[i++], s.m) || !ParseWholeInt(t[i++], s.n) ||
      !ParseWholeInt(t[i++], s.k)) {
    return s;
  }
  if (t[i] == "B") {
    i++;
    if (i >= t.size() || !ParseWholeInt(t[i++], s.b)) {
      return s;
    }
    s.has_b = true;
  }
  if (i >= t.size() || t[i] != "ld") {
    return s;
  }
  i++;
  if (i + 2 >= t.size()) {
    return s;
  }
  if (!ParseWholeInt(t[i++], s.lda) || !ParseWholeInt(t[i++], s.ldb) ||
      !ParseWholeInt(t[i++], s.ldc)) {
    return s;
  }
  for (; i < t.size(); i++) {
    s.suffix += "_" + t[i];
  }
  s.ok = true;
  return s;
}

// True iff `c` can serve `q` by rounding exactly one of {M, N, B} UP to c's value;
// `delta` is the gap, so the caller can pick the closest candidate. Which ld's
// may differ is decided by LdsCompatibleWhen*Varies above, which applies the
// same transpose and tightness rules as ShouldWildcardLd*: an ld that is padded
// rather than tight is not a function of m/n/k, so it must match exactly even
// when the dim it would otherwise track is the one being rounded.
inline bool NearestLargerMatch(const GemmShape& q, const GemmShape& c, int64_t& delta) {
  if (!q.ok || !c.ok) {
    return false;
  }
  if (q.transab != c.transab || q.has_b != c.has_b || q.k != c.k ||
      q.suffix != c.suffix) {
    return false;
  }
  const bool dm = q.m != c.m;
  const bool dn = q.n != c.n;
  const bool db = q.has_b && q.b != c.b;
  if ((dm ? 1 : 0) + (dn ? 1 : 0) + (db ? 1 : 0) != 1) {
    return false;
  }
  const char ta = q.transab[0];
  const char tb = q.transab[1];
  const GemmLeadingDims q_lds{q.lda, q.ldb, q.ldc};
  const GemmLeadingDims c_lds{c.lda, c.ldb, c.ldc};
  int64_t qv = 0;
  int64_t cv = 0;
  if (dm) {
    if (!LdsCompatibleWhenMVaries(ta, q_lds, q.m, c_lds, c.m)) {
      return false;
    }
    qv = q.m;
    cv = c.m;
  } else if (dn) {
    if (!LdsCompatibleWhenNVaries(tb, q_lds, q.n, c_lds, c.n)) {
      return false;
    }
    qv = q.n;
    cv = c.n;
  } else {  // db: no ld depends on B, so all must match.
    if (q.lda != c.lda || q.ldb != c.ldb || q.ldc != c.ldc) {
      return false;
    }
    qv = q.b;
    cv = c.b;
  }
  if (cv < qv) {
    return false;
  }
  delta = cv - qv;
  return true;
}

} // namespace at::cuda::tunable
