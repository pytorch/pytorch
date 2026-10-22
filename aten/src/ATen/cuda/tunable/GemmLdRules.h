#pragma once

#include <cstdint>

// Leading-dimension rules for GEMM signatures, split out of GemmCommon.h so
// they can be unit tested without pulling in ATen or CUDA. Pure integer and
// character logic -- no dependencies beyond <cstdint>.
//
// Two consumers share these rules and must not drift apart:
//   * ShouldWildcardLd* in GemmCommon.h, deciding whether an ld may be replaced
//     by '*' when a logical dim is dynamic.
//   * NearestLargerMatch in Tunable.cpp, deciding whether a tuned entry for one
//     shape may be borrowed for another differing in a single logical dim.

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

} // namespace at::cuda::tunable
