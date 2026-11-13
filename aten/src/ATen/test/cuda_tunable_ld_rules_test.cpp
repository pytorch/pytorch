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
using at::cuda::tunable::GemmShape;
using at::cuda::tunable::IsTight;
using at::cuda::tunable::LdMayVary;
using at::cuda::tunable::LdsCompatibleWhenMVaries;
using at::cuda::tunable::LdsCompatibleWhenNVaries;
using at::cuda::tunable::NearestLargerMatch;
using at::cuda::tunable::ParseGemmShape;
using at::cuda::tunable::SplitUnderscore;
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

TEST(TunableLdRules, IsTightOnlyWhenLdEqualsItsExtent) {
  // The single definition LdMayVary and ShouldWildcardLd* both build on.
  EXPECT_TRUE(IsTight(32, 32));
  EXPECT_FALSE(IsTight(40, 32));  // padded
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

namespace {

// Parse a full params signature the way LookupNearest does.
GemmShape ParseSig(const std::string& sig) {
  return ParseGemmShape(SplitUnderscore(sig));
}

} // namespace

TEST(TunableLdRules, ParsePlainSignature) {
  // Grammar of GemmParams<T>::Signature (GemmCommon.h): ab_M_N_K_ld_lda_ldb_ldc.
  const GemmShape s = ParseSig("NN_1024_512_256_ld_1024_256_1024");
  EXPECT_TRUE(s.ok);
  EXPECT_EQ(s.transab, "NN");
  EXPECT_EQ(s.m, 1024);
  EXPECT_EQ(s.n, 512);
  EXPECT_EQ(s.k, 256);
  EXPECT_FALSE(s.has_b);
  EXPECT_EQ(s.lda, 1024);
  EXPECT_EQ(s.ldb, 256);
  EXPECT_EQ(s.ldc, 1024);
  EXPECT_TRUE(s.suffix.empty());
}

TEST(TunableLdRules, ParseBatchedSignature) {
  // Grammar of GemmStridedBatchedParams<T>::Signature: ab_M_N_K_B_b_ld_... .
  const GemmShape s = ParseSig("NN_8_512_256_B_4_ld_512_256_8");
  EXPECT_TRUE(s.ok);
  EXPECT_TRUE(s.has_b);
  EXPECT_EQ(s.b, 4);
  EXPECT_EQ(s.lda, 512);
}

TEST(TunableLdRules, ParseScaledMmSuffix) {
  // Grammar of ScaledGemmParams<T>::Signature (ROCm): ..._rw_0|1_bias_<dtype>.
  const GemmShape s = ParseSig("NN_1024_512_256_ld_1024_256_1024_rw_1_bias_BFloat16");
  EXPECT_TRUE(s.ok);
  EXPECT_EQ(s.suffix, "_rw_1_bias_BFloat16");
  EXPECT_EQ(s.m, 1024);
}

TEST(TunableLdRules, ParseRejectsWildcards) {
  // Wildcard '*' tokens fail numeric conversion: borrowing is
  // concrete-to-concrete by design, dynamic signatures never parse.
  EXPECT_FALSE(ParseSig("NN_*_512_256_ld_*_256_1024").ok);
  EXPECT_FALSE(ParseSig("NN_1024_512_256_ld_1024_*_1024").ok);
}

TEST(TunableLdRules, ParseRejectsMalformed) {
  EXPECT_FALSE(ParseSig("").ok);
  EXPECT_FALSE(ParseSig("NN_1024").ok);  // too short
  EXPECT_FALSE(ParseSig("N_1024_512_256_ld_1024_256_1024").ok);  // transab != 2
  EXPECT_FALSE(ParseSig("NN_1024_512_256_1024_256_1024").ok);  // missing ld token
  EXPECT_FALSE(ParseSig("NN_x_512_256_ld_1024_256_1024").ok);  // not a number
}

TEST(TunableLdRules, ParseRequiresWholeTokens) {
  // A token has to be consumed entirely, or a malformed signature parses as
  // ok and borrows against a dim it never stated.
  EXPECT_FALSE(ParseSig("NN_12abc_512_256_ld_12_256_12").ok);
  EXPECT_FALSE(ParseSig("NN_1024_512_256_ld_1024_256_1024abc").ok);
  EXPECT_FALSE(ParseSig("NN_+12_512_256_ld_12_256_12").ok);
  EXPECT_FALSE(ParseSig("NN_-12_512_256_ld_12_256_12").ok);
  EXPECT_FALSE(ParseSig("NN__512_256_ld_1024_256_1024").ok);  // empty token
}

TEST(TunableLdRules, NearestMatchRoundsMSingleDimUp) {
  int64_t delta = -1;
  const GemmShape q = ParseSig("NN_32_512_256_ld_32_256_32");
  const GemmShape c = ParseSig("NN_64_512_256_ld_64_256_64");
  EXPECT_TRUE(NearestLargerMatch(q, c, delta));
  EXPECT_EQ(delta, 32);
}

TEST(TunableLdRules, NearestMatchRoundsNUpWhenBNotTransposed) {
  // transb='N': ldb tracks K and is unchanged, lda and ldc never track N, so
  // rounding N up leaves every leading dim identical.
  int64_t delta = -1;
  const GemmShape q = ParseSig("NN_32_512_256_ld_32_256_32");
  const GemmShape c = ParseSig("NN_32_1024_256_ld_32_256_32");
  EXPECT_TRUE(NearestLargerMatch(q, c, delta));
  EXPECT_EQ(delta, 512);
}

TEST(TunableLdRules, NearestMatchRoundsNUpWhenBTransposed) {
  // transb='T': ldb tracks N, so it has to grow with it and be tight in both.
  int64_t delta = -1;
  const GemmShape q = ParseSig("NT_32_16_256_ld_32_16_32");
  EXPECT_TRUE(NearestLargerMatch(q, ParseSig("NT_32_32_256_ld_32_32_32"), delta));
  EXPECT_EQ(delta, 16);
  // Same donor but with ldb padded rather than tight: not a function of N.
  EXPECT_FALSE(
      NearestLargerMatch(q, ParseSig("NT_32_32_256_ld_32_40_32"), delta));
}

TEST(TunableLdRules, NearestMatchRejectsBatchedAgainstNonBatched) {
  // A strided-batched solution and a plain GEMM solution are different ops
  // even at batch 1, so has_b must agree before any dim is compared.
  int64_t delta = -1;
  const GemmShape batched = ParseSig("NN_8_512_256_B_2_ld_512_256_8");
  const GemmShape plain = ParseSig("NN_8_512_256_ld_512_256_8");
  EXPECT_FALSE(NearestLargerMatch(batched, plain, delta));
  EXPECT_FALSE(NearestLargerMatch(plain, batched, delta));
}

TEST(TunableLdRules, NearestMatchRejectsRoundDown) {
  int64_t delta = -1;
  const GemmShape q = ParseSig("NN_64_512_256_ld_64_256_64");
  const GemmShape c = ParseSig("NN_32_512_256_ld_32_256_32");
  EXPECT_FALSE(NearestLargerMatch(q, c, delta));
}

TEST(TunableLdRules, NearestMatchRejectsTwoDimsDiffer) {
  int64_t delta = -1;
  const GemmShape q = ParseSig("NN_32_512_256_ld_32_256_32");
  const GemmShape c = ParseSig("NN_64_1024_256_ld_64_256_64");
  EXPECT_FALSE(NearestLargerMatch(q, c, delta));
}

TEST(TunableLdRules, NearestMatchRejectsKMismatch) {
  // A solution is only valid for the exact K it was tuned at.
  int64_t delta = -1;
  const GemmShape q = ParseSig("NN_32_512_256_ld_32_256_32");
  const GemmShape c = ParseSig("NN_64_512_128_ld_64_128_64");
  EXPECT_FALSE(NearestLargerMatch(q, c, delta));
}

TEST(TunableLdRules, NearestMatchRejectsTransposeAndSuffixMismatch) {
  int64_t delta = -1;
  const GemmShape q = ParseSig("NN_32_512_256_ld_32_256_32");
  EXPECT_FALSE(
      NearestLargerMatch(q, ParseSig("NT_64_512_256_ld_256_256_64"), delta));
  EXPECT_FALSE(NearestLargerMatch(
      q, ParseSig("NN_64_512_256_ld_64_256_64_rw_1_bias_BFloat16"), delta));
}

TEST(TunableLdRules, NearestMatchBatchVariesRequiresLdMatch) {
  // No ld depends on B, so all leading dims must match exactly.
  int64_t delta = -1;
  const GemmShape q = ParseSig("NN_8_512_256_B_2_ld_512_256_8");
  const GemmShape c = ParseSig("NN_8_512_256_B_4_ld_512_256_8");
  EXPECT_TRUE(NearestLargerMatch(q, c, delta));
  EXPECT_EQ(delta, 2);
  EXPECT_FALSE(NearestLargerMatch(
      q, ParseSig("NN_8_512_256_B_4_ld_512_256_16"), delta));
}
