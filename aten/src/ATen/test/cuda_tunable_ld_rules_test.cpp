// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include <gtest/gtest.h>

#include <ATen/cuda/tunable/GemmLdRules.h>

// These rules decide whether a tuned GEMM entry may be borrowed for a shape
// differing in one logical dim. Getting them wrong does not corrupt output --
// a borrowed solution is benchmarked before use -- but a false donor burns a
// compile-time worker and can persist Default instead of a real neighbour.
//
// GemmLdRules.h carries no ATen or CUDA dependency precisely so this can be a
// plain unit test rather than something that needs a GPU.

using at::cuda::tunable::GemmLeadingDims;
using at::cuda::tunable::LdMayVary;
using at::cuda::tunable::LdsCompatibleWhenMVaries;
using at::cuda::tunable::LdsCompatibleWhenNVaries;
using at::cuda::tunable::UsesKForLdb;
using at::cuda::tunable::UsesMForLda;

namespace {

// Tight leading dims for an m x k A, k x n B, m x n C in the BLAS frame:
// lda = m (or k when A is transposed), ldb = k, ldc = m.
GemmLeadingDims Tight(int64_t m, int64_t k, bool transa) {
  return GemmLeadingDims{transa ? k : m, k, m};
}

} // namespace

TEST(TunableLdRules, LdaTracksMOnlyWhenANotTransposed) {
  EXPECT_TRUE(UsesMForLda('N'));
  EXPECT_TRUE(UsesMForLda('n'));
  EXPECT_FALSE(UsesMForLda('T'));
  EXPECT_FALSE(UsesMForLda('t'));
  // The conjugate forms are transposes too: lda tracks K, not M.
  EXPECT_FALSE(UsesMForLda('C'));
  EXPECT_FALSE(UsesMForLda('c'));
}

TEST(TunableLdRules, LdbTracksKOnlyWhenBNotTransposed) {
  EXPECT_TRUE(UsesKForLdb('N'));
  EXPECT_TRUE(UsesKForLdb('n'));
  EXPECT_FALSE(UsesKForLdb('T'));
  EXPECT_FALSE(UsesKForLdb('C'));
  EXPECT_FALSE(UsesKForLdb('c'));
}

TEST(TunableLdRules, LdMayVaryOnlyWhenEqualOrTightInBoth) {
  EXPECT_TRUE(LdMayVary(32, 32, 64, 64));   // tight in both
  EXPECT_TRUE(LdMayVary(48, 32, 48, 64));   // equal, padding irrelevant
  EXPECT_FALSE(LdMayVary(40, 32, 80, 64));  // padded in both, and differing
  EXPECT_FALSE(LdMayVary(32, 32, 80, 64));  // tight in one only
  EXPECT_FALSE(LdMayVary(40, 32, 64, 64));  // tight in the other only
}

TEST(TunableLdRules, MVariesTightNonTransposedBorrows) {
  // transa='N': lda and ldc both track M and are tight, ldb matches.
  const GemmLeadingDims q = Tight(/*m=*/32, /*k=*/64, /*transa=*/false);
  const GemmLeadingDims c = Tight(/*m=*/64, /*k=*/64, /*transa=*/false);
  EXPECT_TRUE(LdsCompatibleWhenMVaries('N', q, 32, c, 64));
}

TEST(TunableLdRules, MVariesRejectsConjugateTransposeWithDifferingLda) {
  // transa='C': lda tracks K, so it must match exactly even though M varies.
  // This is the case a bare 't'/'T' check misses.
  const GemmLeadingDims q{/*lda=*/64, /*ldb=*/64, /*ldc=*/32};
  const GemmLeadingDims c{/*lda=*/72, /*ldb=*/64, /*ldc=*/64};
  EXPECT_FALSE(LdsCompatibleWhenMVaries('C', q, 32, c, 64));
  EXPECT_FALSE(LdsCompatibleWhenMVaries('c', q, 32, c, 64));
  // Same shapes but with A merely transposed behave identically.
  EXPECT_FALSE(LdsCompatibleWhenMVaries('T', q, 32, c, 64));
}

TEST(TunableLdRules, MVariesAcceptsConjugateTransposeWithMatchingLda) {
  const GemmLeadingDims q{/*lda=*/64, /*ldb=*/64, /*ldc=*/32};
  const GemmLeadingDims c{/*lda=*/64, /*ldb=*/64, /*ldc=*/64};
  EXPECT_TRUE(LdsCompatibleWhenMVaries('C', q, 32, c, 64));
}

TEST(TunableLdRules, MVariesRejectsPaddedLdc) {
  // ldc tracks M but is padded in both, so it is not a function of M.
  // 32x16 and 64x16 sub-matrices of wider buffers: valid layouts, not donors.
  const GemmLeadingDims q{/*lda=*/32, /*ldb=*/64, /*ldc=*/40};
  const GemmLeadingDims c{/*lda=*/64, /*ldb=*/64, /*ldc=*/80};
  EXPECT_FALSE(LdsCompatibleWhenMVaries('N', q, 32, c, 64));
}

TEST(TunableLdRules, MVariesRejectsPaddedLda) {
  const GemmLeadingDims q{/*lda=*/40, /*ldb=*/64, /*ldc=*/32};
  const GemmLeadingDims c{/*lda=*/72, /*ldb=*/64, /*ldc=*/64};
  EXPECT_FALSE(LdsCompatibleWhenMVaries('N', q, 32, c, 64));
}

TEST(TunableLdRules, MVariesRequiresLdbToMatch) {
  // Nothing about ldb depends on M.
  const GemmLeadingDims q{/*lda=*/32, /*ldb=*/64, /*ldc=*/32};
  const GemmLeadingDims c{/*lda=*/64, /*ldb=*/96, /*ldc=*/64};
  EXPECT_FALSE(LdsCompatibleWhenMVaries('N', q, 32, c, 64));
}

TEST(TunableLdRules, NVariesTransposedBTightBorrows) {
  // transb='T': ldb tracks N. Tight in both, lda and ldc match.
  const GemmLeadingDims q{/*lda=*/32, /*ldb=*/16, /*ldc=*/32};
  const GemmLeadingDims c{/*lda=*/32, /*ldb=*/32, /*ldc=*/32};
  EXPECT_TRUE(LdsCompatibleWhenNVaries('T', q, 16, c, 32));
}

TEST(TunableLdRules, NVariesRejectsPaddedLdb) {
  const GemmLeadingDims q{/*lda=*/32, /*ldb=*/24, /*ldc=*/32};
  const GemmLeadingDims c{/*lda=*/32, /*ldb=*/40, /*ldc=*/32};
  EXPECT_FALSE(LdsCompatibleWhenNVaries('T', q, 16, c, 32));
}

TEST(TunableLdRules, NVariesRequiresLdbToMatchWhenBNotTransposed) {
  // transb='N': ldb tracks K, which is unchanged, so it must match exactly.
  const GemmLeadingDims q{/*lda=*/32, /*ldb=*/16, /*ldc=*/32};
  const GemmLeadingDims c{/*lda=*/32, /*ldb=*/32, /*ldc=*/32};
  EXPECT_FALSE(LdsCompatibleWhenNVaries('N', q, 16, c, 32));
  const GemmLeadingDims same{/*lda=*/32, /*ldb=*/16, /*ldc=*/32};
  EXPECT_TRUE(LdsCompatibleWhenNVaries('N', q, 16, same, 32));
}

TEST(TunableLdRules, NVariesRequiresLdaAndLdcToMatch) {
  // Neither lda nor ldc depends on N.
  const GemmLeadingDims q{/*lda=*/32, /*ldb=*/16, /*ldc=*/32};
  EXPECT_FALSE(LdsCompatibleWhenNVaries(
      'T', q, 16, GemmLeadingDims{64, 32, 32}, 32));
  EXPECT_FALSE(LdsCompatibleWhenNVaries(
      'T', q, 16, GemmLeadingDims{32, 32, 64}, 32));
}
