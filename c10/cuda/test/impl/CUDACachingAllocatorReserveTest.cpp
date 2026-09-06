#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

using c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig;
using c10::cuda::CUDACachingAllocator::ExpandableSegmentReserveDecision;
using c10::cuda::CUDACachingAllocator::
    getExpandableSegmentReserveClassForStream;
using c10::cuda::CUDACachingAllocator::
    setDefaultExpandableSegmentReserveFractionForClass;
using c10::cuda::CUDACachingAllocator::
    setExpandableSegmentReserveClassForStream;

namespace {
constexpr size_t kTotal = size_t{100} << 30; // 100 GiB device
constexpr size_t kGiB = size_t{1} << 30;
// Sentinel for "no reserve configured" that no test expects as a real reserve
// (a configured reserve is always > 0). Using value_or keeps optional access
// total, which clang-tidy's bugprone-unchecked-optional-access requires.
constexpr size_t kUnset = 0;

void setConf(const std::string& conf) {
  c10::CachingAllocator::setAllocatorSettings(conf);
}

// The reserve bytes a class resolves to, via the production snapshot API;
// kUnset when no reserve is configured for the class.
size_t reserveBytes(const std::string& reserve_class) {
  return CUDAAllocatorConfig::expandable_segments_reserve_decision(
             reserve_class, kTotal)
      .reserve_bytes.value_or(kUnset);
}
} // namespace

// Note: the "unset => nullopt" no-op path (keeps the historical 9/8 reserve
// byte-for-byte) is guaranteed structurally by
// m_expandable_segments_reserve_set defaulting to false; it is not unit-tested
// here because the config singleton cannot be reset to its unset state once
// another test sets a reserve.

TEST(ExpandableSegmentReserveTest, DefaultFractionAndGiB) {
  setConf("expandable_segments_reserve:0.5");
  EXPECT_EQ(reserveBytes(""), kTotal / 2);

  setConf("expandable_segments_reserve:40G");
  EXPECT_EQ(reserveBytes("any"), 40 * kGiB);
}

TEST(ExpandableSegmentReserveTest, PerClassOverrideAndFallback) {
  setConf(
      "expandable_segments_reserve:0.9,"
      "expandable_segments_reserve_by_class:[serving:0.25,ads_sparse:8G]");
  EXPECT_EQ(reserveBytes("serving"), kTotal / 4);
  EXPECT_EQ(reserveBytes("ads_sparse"), 8 * kGiB);
  // Unknown class falls back to the default reserve (0.9 of total).
  EXPECT_EQ(reserveBytes("other"), (kTotal / 10) * 9);
  EXPECT_TRUE(CUDAAllocatorConfig::expandable_segments_reserve_decision(
                  "serving", kTotal)
                  .class_known);
  EXPECT_FALSE(
      CUDAAllocatorConfig::expandable_segments_reserve_decision("other", kTotal)
          .class_known);
}

TEST(ExpandableSegmentReserveTest, ReserveDecisionSnapshot) {
  setConf(
      "expandable_segments_reserve:0.9,"
      "expandable_segments_reserve_by_class:[serving:0.25]");
  auto known = CUDAAllocatorConfig::expandable_segments_reserve_decision(
      "serving", kTotal);
  EXPECT_TRUE(known.reserve_bytes.has_value());
  EXPECT_EQ(known.reserve_bytes.value_or(kUnset), kTotal / 4);
  EXPECT_TRUE(known.class_known);
  // Unknown class falls back to the global default (0.9) and is not "known".
  auto fallback = CUDAAllocatorConfig::expandable_segments_reserve_decision(
      "other", kTotal);
  EXPECT_TRUE(fallback.reserve_bytes.has_value());
  EXPECT_EQ(fallback.reserve_bytes.value_or(kUnset), (kTotal / 10) * 9);
  EXPECT_FALSE(fallback.class_known);
}

TEST(ExpandableSegmentReserveTest, HugeValueSaturatesInsteadOfUB) {
  // A pathological config must saturate at SIZE_MAX rather than invoke UB in
  // the double->size_t narrowing (1e20 GiB far exceeds SIZE_MAX). The saturated
  // value is bounded later by clamp_reserve_bytes (see ClampReserveBytes).
  setConf("expandable_segments_reserve:1e20G");
  EXPECT_EQ(reserveBytes(""), std::numeric_limits<size_t>::max());
}

TEST(ExpandableSegmentReserveTest, InvalidValuesAreIgnoredNotFatal) {
  // Establish a known-good default.
  setConf("expandable_segments_reserve:0.5");
  ASSERT_EQ(reserveBytes(""), kTotal / 2);
  // Malformed / out-of-range values must not throw or abort; they are logged
  // and ignored, leaving the previously-set value intact.
  EXPECT_NO_THROW(setConf("expandable_segments_reserve:abc"));
  EXPECT_NO_THROW(setConf("expandable_segments_reserve:G"));
  EXPECT_NO_THROW(setConf("expandable_segments_reserve:-1"));
  EXPECT_NO_THROW(setConf("expandable_segments_reserve:0"));
  EXPECT_EQ(reserveBytes(""), kTotal / 2);
  // A malformed per-class entry is skipped; valid entries in the same list
  // still apply.
  EXPECT_NO_THROW(
      setConf("expandable_segments_reserve_by_class:[serving:0.25,bad:12X]"));
  EXPECT_EQ(reserveBytes("serving"), kTotal / 4);
  EXPECT_FALSE(
      CUDAAllocatorConfig::expandable_segments_reserve_decision("bad", kTotal)
          .class_known);
}

TEST(ExpandableSegmentReserveTest, ClampReserveBytes) {
  // Pure clamp arithmetic (the production ExpandableSegment ctor path), tested
  // without a device. full_reserve models 9/8 of an ~80 GiB GPU.
  constexpr size_t kFull = size_t{90} << 30;

  // No override configured -> the full reserve is kept unchanged.
  ExpandableSegmentReserveDecision none;
  EXPECT_EQ(CUDAAllocatorConfig::clamp_reserve_bytes(none, kFull), kFull);

  // Configured below full -> used as-is.
  ExpandableSegmentReserveDecision below;
  below.reserve_bytes = 40 * kGiB;
  EXPECT_EQ(CUDAAllocatorConfig::clamp_reserve_bytes(below, kFull), 40 * kGiB);

  // Configured above full -> capped at full.
  ExpandableSegmentReserveDecision above;
  above.reserve_bytes = kFull + (32 * kGiB);
  EXPECT_EQ(CUDAAllocatorConfig::clamp_reserve_bytes(above, kFull), kFull);

  // Saturated reserve (pathological config) -> capped at full, never SIZE_MAX,
  // so the reserve fed to numSegments() cannot overflow to 0.
  ExpandableSegmentReserveDecision saturated;
  saturated.reserve_bytes = std::numeric_limits<size_t>::max();
  EXPECT_EQ(CUDAAllocatorConfig::clamp_reserve_bytes(saturated, kFull), kFull);
}

TEST(ExpandableSegmentReserveTest, StreamReserveClassTagRoundTrip) {
  // The tag map is keyed by the raw cudaStream_t handle, so a fake, never
  // dereferenced handle exercises set/get without a device. Take the address of
  // a local (a distinct, non-null pointer) rather than casting an integer,
  // which clang-tidy's performance-no-int-to-ptr flags.
  char storage = 0;
  const auto stream = reinterpret_cast<cudaStream_t>(&storage);

  // Untagged handle -> empty class.
  EXPECT_EQ(getExpandableSegmentReserveClassForStream(stream), "");
  // Tag, then read back.
  setExpandableSegmentReserveClassForStream(stream, "serving");
  EXPECT_EQ(getExpandableSegmentReserveClassForStream(stream), "serving");
  // Overwrite.
  setExpandableSegmentReserveClassForStream(stream, "lrm_sparse");
  EXPECT_EQ(getExpandableSegmentReserveClassForStream(stream), "lrm_sparse");
  // Clear via the empty class.
  setExpandableSegmentReserveClassForStream(stream, "");
  EXPECT_EQ(getExpandableSegmentReserveClassForStream(stream), "");
}

TEST(ExpandableSegmentReserveTest, SetDefaultReserveFractionPrecedence) {
  // An explicit per-class config entry wins over a code-side default.
  setConf("expandable_segments_reserve_by_class:[serving:0.25]");
  setDefaultExpandableSegmentReserveFractionForClass("serving", 0.9);
  EXPECT_EQ(reserveBytes("serving"), kTotal / 4);

  // Once a global reserve is set via config, a code-side default is a no-op: an
  // unseeded class falls back to the global default, not the code-side
  // fraction.
  setConf("expandable_segments_reserve:0.5");
  setDefaultExpandableSegmentReserveFractionForClass("telemetry", 0.9);
  auto d = CUDAAllocatorConfig::expandable_segments_reserve_decision(
      "telemetry", kTotal);
  EXPECT_FALSE(d.class_known);
  EXPECT_EQ(d.reserve_bytes.value_or(kUnset), kTotal / 2);
}
