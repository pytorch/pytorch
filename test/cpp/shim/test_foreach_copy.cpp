// Checks aoti_torch__foreach_copy_, which every build ships.

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <vector>

namespace {

std::vector<AtenTensorHandle> handles(std::vector<at::Tensor>& tensors) {
  std::vector<AtenTensorHandle> result;
  for (auto& t : tensors) {
    result.push_back(torch::aot_inductor::tensor_pointer_to_tensor_handle(&t));
  }
  return result;
}

// Copies lists of one dtype and layout, which the CUDA kernel fuses, then mixed
// ones, which it copies one by one.
void check_copies(c10::Device device) {
  auto floats = at::TensorOptions().device(device);
  auto longs = floats.dtype(at::kLong);
  for (bool uniform : {true, false}) {
    std::vector<at::Tensor> dst = {
        at::empty({4}, floats), at::empty({3, 2}, floats)};
    std::vector<at::Tensor> src = {
        at::randn({4}, floats), at::randn({3, 2}, floats)};
    if (!uniform) {
      dst[1] = at::empty({2, 3}, floats).t();
      dst.push_back(at::empty({5}, longs));
      src.push_back(at::randint(100, {5}, longs));
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

TEST(AOTIForeachCopyTest, CopiesEachTensor) {
  check_copies(at::kCPU);
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
  for (int64_t n : {0, -1}) {
    EXPECT_EQ(
        aoti_torch__foreach_copy_(
            dst_handles.data(), n, src_handles.data(), n, 0),
        AOTI_TORCH_FAILURE);
  }
  dst[0] = at::empty({3});
  EXPECT_EQ(
      aoti_torch__foreach_copy_(
          dst_handles.data(), 2, src_handles.data(), 2, 0),
      AOTI_TORCH_FAILURE);
}

TEST(AOTIForeachCopyCudaTest, CopiesDeviceTensors) {
  if (!at::hasCUDA()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  check_copies(at::kCUDA);
}

TEST(AOTIForeachCopyCudaTest, CopiesFromPinnedMemoryAsynchronously) {
  if (!at::hasCUDA()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  std::vector<at::Tensor> host = {
      at::randn({4}).pin_memory(), at::randn({3, 2}).pin_memory()};
  std::vector<at::Tensor> device = {
      at::empty({4}, at::kCUDA), at::empty({3, 2}, at::kCUDA)};
  auto host_handles = handles(host);
  auto device_handles = handles(device);
  ASSERT_EQ(
      aoti_torch__foreach_copy_(
          device_handles.data(), 2, host_handles.data(), 2, 1),
      AOTI_TORCH_SUCCESS);
  // cpu() copies on the same stream, after the asynchronous copies.
  for (size_t i = 0; i < host.size(); ++i) {
    EXPECT_TRUE(device[i].cpu().equal(host[i]));
  }
}

} // namespace
