// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Tests for TensorMetadata::dynamic_check and ParameterMetadata::dynamic_check,
// which enable a single compiled AOTI kernel to serve multiple input shapes by
// matching dtype/device/rank instead of exact sizes and strides.

#if !defined(C10_MOBILE) && !defined(ANDROID)

#include <gtest/gtest.h>

#include <torch/csrc/inductor/aoti_eager/kernel_meta_info.h>

namespace torch::inductor {

namespace {

TensorMetadata makeTensorMeta(
    c10::ScalarType dtype,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides,
    c10::Device device = c10::Device(c10::kCPU)) {
  return TensorMetadata(
      /*is_symbolic=*/false,
      dtype,
      device,
      c10::DispatchKeySet(c10::DispatchKey::CPU),
      std::move(sizes),
      std::move(strides));
}

} // namespace

// ---------- TensorMetadata::dynamic_check ----------

TEST(TensorMetadataDynamicCheckTest, MatchesSameDtypeDeviceRankDifferentSizes) {
  auto a = makeTensorMeta(c10::kFloat, {4, 8}, {8, 1});
  auto b = makeTensorMeta(c10::kFloat, {16, 32}, {32, 1});
  EXPECT_TRUE(a.dynamic_check(b));
  // Exact comparison should fail because sizes differ
  EXPECT_FALSE(a == b);
}

TEST(TensorMetadataDynamicCheckTest, RejectsDifferentDtype) {
  auto a = makeTensorMeta(c10::kFloat, {4, 8}, {8, 1});
  auto b = makeTensorMeta(c10::kHalf, {4, 8}, {8, 1});
  EXPECT_FALSE(a.dynamic_check(b));
}

TEST(TensorMetadataDynamicCheckTest, RejectsDifferentRank) {
  auto a = makeTensorMeta(c10::kFloat, {4, 8}, {8, 1});
  auto b = makeTensorMeta(c10::kFloat, {4, 8, 2}, {16, 2, 1});
  EXPECT_FALSE(a.dynamic_check(b));
}

TEST(TensorMetadataDynamicCheckTest, RejectsDifferentDevice) {
  auto cpu = makeTensorMeta(c10::kFloat, {4}, {1}, c10::Device(c10::kCPU));
  auto cuda = makeTensorMeta(c10::kFloat, {4}, {1}, c10::Device(c10::kCUDA, 0));
  EXPECT_FALSE(cpu.dynamic_check(cuda));
}

TEST(TensorMetadataDynamicCheckTest, MatchesIdenticalMetadata) {
  auto a = makeTensorMeta(c10::kFloat, {4, 8}, {8, 1});
  auto b = makeTensorMeta(c10::kFloat, {4, 8}, {8, 1});
  EXPECT_TRUE(a.dynamic_check(b));
  EXPECT_TRUE(a == b);
}

TEST(TensorMetadataDynamicCheckTest, MatchesScalarTensors) {
  auto a = makeTensorMeta(c10::kFloat, {}, {});
  auto b = makeTensorMeta(c10::kFloat, {}, {});
  EXPECT_TRUE(a.dynamic_check(b));
}

// ---------- ParameterMetadata::dynamic_check ----------

TEST(ParameterMetadataDynamicCheckTest, TensorMatchesDifferentSizes) {
  ParameterMetadata a(makeTensorMeta(c10::kFloat, {4, 8}, {8, 1}), 0);
  ParameterMetadata b(makeTensorMeta(c10::kFloat, {16, 32}, {32, 1}), 0);
  EXPECT_TRUE(a.dynamic_check(b));
  EXPECT_FALSE(a == b);
}

TEST(ParameterMetadataDynamicCheckTest, TensorRejectsDifferentDtype) {
  ParameterMetadata a(makeTensorMeta(c10::kFloat, {4}, {1}), 0);
  ParameterMetadata b(makeTensorMeta(c10::kHalf, {4}, {1}), 0);
  EXPECT_FALSE(a.dynamic_check(b));
}

TEST(ParameterMetadataDynamicCheckTest, RejectsDifferentOrder) {
  ParameterMetadata a(makeTensorMeta(c10::kFloat, {4}, {1}), 0);
  ParameterMetadata b(makeTensorMeta(c10::kFloat, {4}, {1}), 1);
  EXPECT_FALSE(a.dynamic_check(b));
}

TEST(ParameterMetadataDynamicCheckTest, ScalarUsesExactMatch) {
  ParameterMetadata a(c10::Scalar(1.5), 0);
  ParameterMetadata b(c10::Scalar(1.5), 0);
  EXPECT_TRUE(a.dynamic_check(b));

  ParameterMetadata c(c10::Scalar(2.0), 0);
  EXPECT_FALSE(a.dynamic_check(c));
}

TEST(ParameterMetadataDynamicCheckTest, TensorListMatchesDifferentSizes) {
  std::vector<TensorMetadata> list_a = {
      makeTensorMeta(c10::kFloat, {4, 8}, {8, 1}),
      makeTensorMeta(c10::kFloat, {2, 8}, {8, 1}),
  };
  std::vector<TensorMetadata> list_b = {
      makeTensorMeta(c10::kFloat, {16, 32}, {32, 1}),
      makeTensorMeta(c10::kFloat, {8, 32}, {32, 1}),
  };
  ParameterMetadata a(list_a, 0);
  ParameterMetadata b(list_b, 0);
  EXPECT_TRUE(a.dynamic_check(b));
}

TEST(ParameterMetadataDynamicCheckTest, TensorListRejectsDifferentLengths) {
  std::vector<TensorMetadata> one = {makeTensorMeta(c10::kFloat, {4}, {1})};
  std::vector<TensorMetadata> two = {
      makeTensorMeta(c10::kFloat, {4}, {1}),
      makeTensorMeta(c10::kFloat, {8}, {1}),
  };
  ParameterMetadata a(one, 0);
  ParameterMetadata b(two, 0);
  EXPECT_FALSE(a.dynamic_check(b));
}

TEST(ParameterMetadataDynamicCheckTest, StringUsesExactMatch) {
  ParameterMetadata a(std::string("sum"), 0);
  ParameterMetadata b(std::string("sum"), 0);
  EXPECT_TRUE(a.dynamic_check(b));

  ParameterMetadata c(std::string("mean"), 0);
  EXPECT_FALSE(a.dynamic_check(c));
}

} // namespace torch::inductor

#endif // !defined(C10_MOBILE) && !defined(ANDROID)
