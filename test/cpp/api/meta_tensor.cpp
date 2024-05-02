#include <gtest/gtest.h>

#include <ATen/MetaFunctions.h>
#include <torch/torch.h>

#include <vector>

TEST(MetaTensorTest, MetaDeviceApi) {
  auto a = at::ones({4}, at::kFloat);
  auto b = at::ones({3, 4}, at::kFloat);
  // at::add() will return a meta tensor if its inputs are also meta tensors.
  auto out_meta = at::add(a.to(c10::kMeta), b.to(c10::kMeta));

  ASSERT_EQ(a.device(), c10::kCPU);
  ASSERT_EQ(b.device(), c10::kCPU);
  ASSERT_EQ(out_meta.device(), c10::kMeta);
  c10::IntArrayRef sizes_actual = out_meta.sizes();
  std::vector<int64_t> sizes_expected = std::vector<int64_t>{3, 4};
  ASSERT_EQ(sizes_actual, sizes_expected);
}

TEST(MetaTensorTest, MetaNamespaceApi) {
  auto a = at::ones({4}, at::kFloat);
  auto b = at::ones({3, 4}, at::kFloat);
  // The at::meta:: namespace take in tensors from any backend
  // and return a meta tensor.
  auto out_meta = at::meta::add(a, b);

  ASSERT_EQ(a.device(), c10::kCPU);
  ASSERT_EQ(b.device(), c10::kCPU);
  ASSERT_EQ(out_meta.device(), c10::kMeta);
  c10::IntArrayRef sizes_actual = out_meta.sizes();
  std::vector<int64_t> sizes_expected = std::vector<int64_t>{3, 4};
  ASSERT_EQ(sizes_actual, sizes_expected);
}
