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
// These rules must not drift from ShouldWildcardLd* in GemmCommon.h, which
// applies the same principle -- a padded ld is not a function of m/n/k -- to a
// single shape, where the functions here compare two.
//
// Also hosts the signature parser and borrow check used by
// TuningResultsManager::LookupNearest, so the grammar sits next to the rules.

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

// "Tight" means the leading dim equals its logical extent, so it carries no
// independent padding and is a function of m/n/k. The one definition: both
// LdMayVary below and ShouldWildcardLd* in GemmCommon.h build on it, so the
// two sites cannot drift.
inline bool IsTight(int64_t ld, int64_t dim) {
  return ld == dim;
}

// A leading dim that tracks the varying logical dim may differ between the two
// shapes only when it is tight in BOTH: a padded ld is not a function of
// m/n/k, so treating it as one would pair layouts with different padding.
inline bool LdMayVary(int64_t q_ld, int64_t q_dim, int64_t c_ld, int64_t c_dim) {
  return q_ld == c_ld || (IsTight(q_ld, q_dim) && IsTight(c_ld, c_dim));
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

// Parse a token that must be a whole non-negative decimal integer. Rejects
// prefix matches ("12abc"), signs and empties, so a malformed signature cannot
// parse as ok. Must not throw either: every wildcard entry in the results map
// reaches this on every LookupNearest call.
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
