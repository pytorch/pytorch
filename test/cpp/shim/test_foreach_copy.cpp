// Checks aoti_torch__foreach_copy_, which every build ships.

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <vector>

namespace {

std::vector<AtenTensorHandle> handles(std::vector<at::Tensor>& tensors) {
  std::vector<AtenTensorHandle> result;
  for (auto& t : tensors) {
    result.push_back(reinterpret_cast<AtenTensorHandle>(&t));
  }
  return result;
}

TEST(AOTIForeachCopyTest, CopiesEachTensor) {
  // Lists of one dtype and layout, which the CUDA kernel fuses, then mixed
  // ones, which it doesn't.
  for (bool uniform : {true, false}) {
    std::vector<at::Tensor> dst = {at::empty({4}), at::empty({3, 2})};
    std::vector<at::Tensor> src = {at::randn({4}), at::randn({3, 2})};
    if (!uniform) {
      dst[1] = at::empty({2, 3}).t();
      dst.push_back(at::empty({5}, at::kLong));
      src.push_back(at::randint(100, {5}, at::kLong));
    }
    auto dst_handles = handles(dst);
    auto src_handles = handles(src);
    auto n = static_cast<int64_t>(dst.size());
    ASSERT_EQ(
        aoti_torch__foreach_copy_(
            dst_handles.data(), n, src_handles.data(), n, 0),
        AOTI_TORCH_SUCCESS);
    for (size_t i = 0; i < dst.size(); ++i) {
      EXPECT_TRUE(dst[i].equal(src[i]));
    }
  }
}

TEST(AOTIForeachCopyTest, ChecksItsArguments) {
  std::vector<at::Tensor> dst = {at::empty({4}), at::empty({3, 2})};
  std::vector<at::Tensor> src = {at::randn({4}), at::randn({3, 2})};
  auto dst_handles = handles(dst);
  auto src_handles = handles(src);
  EXPECT_EQ(
      aoti_torch__foreach_copy_(
          dst_handles.data(), 1, src_handles.data(), 2, 0),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(
      aoti_torch__foreach_copy_(nullptr, 1, nullptr, 1, 0), AOTI_TORCH_FAILURE);
  dst[0] = at::empty({3});
  EXPECT_EQ(
      aoti_torch__foreach_copy_(
          dst_handles.data(), 2, src_handles.data(), 2, 0),
      AOTI_TORCH_FAILURE);
}

} // namespace
