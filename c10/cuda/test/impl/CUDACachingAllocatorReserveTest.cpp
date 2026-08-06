#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDAAllocatorConfig.h>

#include <gtest/gtest.h>

#include <limits>

using c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig;

namespace {
constexpr size_t kTotal = size_t{100} << 30; // 100 GiB device
constexpr size_t kGiB = size_t{1} << 30;

void setConf(const std::string& conf) {
  c10::CachingAllocator::setAllocatorSettings(conf);
}
} // namespace

// Note: the "unset => nullopt" no-op path (keeps the historical 9/8 reserve
// byte-for-byte) is guaranteed structurally by
// m_expandable_segments_reserve_set defaulting to false; it is not unit-tested
// here because the config singleton cannot be reset to its unset state once
// another test sets a reserve.

TEST(ExpandableSegmentReserveTest, DefaultFractionAndGiB) {
  setConf("expandable_segments_reserve:0.5");
  auto frac =
      CUDAAllocatorConfig::expandable_segments_reserve_bytes("", kTotal);
  ASSERT_TRUE(frac.has_value());
  EXPECT_EQ(*frac, kTotal / 2);

  setConf("expandable_segments_reserve:40G");
  auto gib =
      CUDAAllocatorConfig::expandable_segments_reserve_bytes("any", kTotal);
  ASSERT_TRUE(gib.has_value());
  EXPECT_EQ(*gib, 40 * kGiB);
}

TEST(ExpandableSegmentReserveTest, PerClassOverrideAndFallback) {
  setConf(
      "expandable_segments_reserve:0.9,"
      "expandable_segments_reserve_by_class:[serving:0.25,ads_sparse:8G]");
  EXPECT_EQ(
      *CUDAAllocatorConfig::expandable_segments_reserve_bytes(
          "serving", kTotal),
      kTotal / 4);
  EXPECT_EQ(
      *CUDAAllocatorConfig::expandable_segments_reserve_bytes(
          "ads_sparse", kTotal),
      8 * kGiB);
  // Unknown class falls back to the default reserve (0.9 of total).
  EXPECT_EQ(
      *CUDAAllocatorConfig::expandable_segments_reserve_bytes("other", kTotal),
      (kTotal / 10) * 9);
  EXPECT_TRUE(
      CUDAAllocatorConfig::expandable_segments_has_reserve_class("serving"));
  EXPECT_FALSE(
      CUDAAllocatorConfig::expandable_segments_has_reserve_class("other"));
}

TEST(ExpandableSegmentReserveTest, MinReserveOverride) {
  // The 16 GiB default is a construction-time default; it is not asserted here
  // because the config singleton cannot be reset between tests (a prior test
  // may have set min_reserve). Verify explicit overrides in both GiB and
  // fraction.
  setConf("expandable_segments_min_reserve:32G");
  EXPECT_EQ(
      CUDAAllocatorConfig::expandable_segments_min_reserve_bytes(kTotal),
      32 * kGiB);
  setConf("expandable_segments_min_reserve:0.1");
  EXPECT_EQ(
      CUDAAllocatorConfig::expandable_segments_min_reserve_bytes(kTotal),
      kTotal / 10);
}

TEST(ExpandableSegmentReserveTest, ReserveDecisionSnapshot) {
  setConf(
      "expandable_segments_reserve:0.9,"
      "expandable_segments_reserve_by_class:[serving:0.25]");
  auto known = CUDAAllocatorConfig::expandable_segments_reserve_decision(
      "serving", kTotal);
  ASSERT_TRUE(known.reserve_bytes.has_value());
  EXPECT_EQ(*known.reserve_bytes, kTotal / 4);
  EXPECT_TRUE(known.class_known);
  // Unknown class falls back to the global default (0.9) and is not "known".
  auto fallback = CUDAAllocatorConfig::expandable_segments_reserve_decision(
      "other", kTotal);
  ASSERT_TRUE(fallback.reserve_bytes.has_value());
  EXPECT_EQ(*fallback.reserve_bytes, (kTotal / 10) * 9);
  EXPECT_FALSE(fallback.class_known);
}

TEST(ExpandableSegmentReserveTest, HugeValueSaturatesInsteadOfUB) {
  // A pathological config must saturate at SIZE_MAX rather than invoke UB in
  // the double->size_t narrowing (1e20 GiB far exceeds SIZE_MAX).
  setConf("expandable_segments_reserve:1e20G");
  auto b = CUDAAllocatorConfig::expandable_segments_reserve_bytes("", kTotal);
  ASSERT_TRUE(b.has_value());
  EXPECT_EQ(*b, std::numeric_limits<size_t>::max());
}

TEST(ExpandableSegmentReserveTest, InvalidValuesAreIgnoredNotFatal) {
  // Establish a known-good default.
  setConf("expandable_segments_reserve:0.5");
  ASSERT_EQ(
      *CUDAAllocatorConfig::expandable_segments_reserve_bytes("", kTotal),
      kTotal / 2);
  // Malformed / out-of-range values must not throw or abort; they are logged
  // and ignored, leaving the previously-set value intact.
  EXPECT_NO_THROW(setConf("expandable_segments_reserve:abc"));
  EXPECT_NO_THROW(setConf("expandable_segments_reserve:G"));
  EXPECT_NO_THROW(setConf("expandable_segments_reserve:-1"));
  EXPECT_NO_THROW(setConf("expandable_segments_reserve:0"));
  EXPECT_EQ(
      *CUDAAllocatorConfig::expandable_segments_reserve_bytes("", kTotal),
      kTotal / 2);
  // A malformed per-class entry is skipped; valid entries in the same list
  // still apply.
  EXPECT_NO_THROW(
      setConf("expandable_segments_reserve_by_class:[serving:0.25,bad:12X]"));
  EXPECT_EQ(
      *CUDAAllocatorConfig::expandable_segments_reserve_bytes(
          "serving", kTotal),
      kTotal / 4);
  EXPECT_FALSE(
      CUDAAllocatorConfig::expandable_segments_has_reserve_class("bad"));
}
