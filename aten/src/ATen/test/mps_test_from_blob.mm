#include <ATen/mps/MPSAllocator.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

namespace {

#define ASSERT_TENSORS_EQ(cpu_tensor, mps_tensor)                                     \
  {                                                                                   \
    auto const_cpu_tensor_ptr = cpu_tensor.to(torch::kCPU).const_data_ptr<uint8_t>(); \
    auto const_mps_tensor_ptr = mps_tensor.to(torch::kCPU).const_data_ptr<uint8_t>(); \
    for (uint32_t i = 0; i < kTestBufferSize; ++i) {                                  \
      EXPECT_EQ(const_cpu_tensor_ptr[i], const_mps_tensor_ptr[i]) << "I: " << i;      \
    }                                                                                 \
  }

#define ASSERT_TENSORS_EQ_AND_PROPERTIES_UNCHANGED(cpu_tensor, mps_tensor) \
  {                                                                        \
    ASSERT_TENSORS_EQ(cpu_tensor, mps_tensor);                             \
    ASSERT_TRUE(cpu_tensor.is_cpu());                                      \
    ASSERT_TRUE(cpu_tensor.dtype() == torch::kUInt8);                      \
    ASSERT_TRUE(mps_tensor.is_mps());                                      \
    ASSERT_TRUE(mps_tensor.dtype() == torch::kUInt8);                      \
  }

} // namespace

TEST(MPSTestFromBlob, SharedMTLBufferFromBlob) {
  // fail if mps isn't available
  ASSERT_TRUE(torch::mps::is_available());

  constexpr uint32_t kTestBufferSize = 10000;

  auto alloc = at::mps::getIMPSAllocator(true);

  c10::DataPtr raw_buffer = alloc->allocate(kTestBufferSize);
  id<MTLBuffer> buffer = __builtin_bit_cast(id<MTLBuffer>, raw_buffer.mutable_get());

  // An example of editing raw buffer, and sync between CPU/GPU
  std::memset(buffer.contents, 0xCC, kTestBufferSize);
  torch::mps::synchronize();

  // Create a CPU tensor from blob.
  auto cpu_options =
      torch::TensorOptions().dtype(torch::kUInt8).layout(torch::kStrided).device(torch::kCPU).requires_grad(false);
  torch::Tensor cpu_tensor = torch::from_blob(buffer.contents, {kTestBufferSize}, {1}, cpu_options);

  // Create a MPS tensor from blob.
  auto mps_options =
      torch::TensorOptions().dtype(torch::kUInt8).layout(torch::kStrided).device(torch::kMPS).requires_grad(false);
  torch::Tensor mps_tensor = torch::from_blob(buffer, {kTestBufferSize}, {1}, mps_options);

  ASSERT_TENSORS_EQ_AND_PROPERTIES_UNCHANGED(cpu_tensor, mps_tensor);

  // Modify the cpu tensor.
  cpu_tensor -= 3;
  torch::mps::synchronize();
  ASSERT_TENSORS_EQ_AND_PROPERTIES_UNCHANGED(cpu_tensor, mps_tensor);

  // Modify the mps tensor
  mps_tensor += 5;
  torch::mps::synchronize();
  ASSERT_TENSORS_EQ_AND_PROPERTIES_UNCHANGED(cpu_tensor, mps_tensor);

  const torch::Tensor cpu_random =
      torch::randint(255, {kTestBufferSize}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));

  std::memset(buffer.contents, 0, kTestBufferSize);
  cpu_tensor += cpu_random;
  ASSERT_TENSORS_EQ(cpu_tensor, cpu_random);
  torch::mps::synchronize();
  ASSERT_TENSORS_EQ_AND_PROPERTIES_UNCHANGED(cpu_tensor, mps_tensor);

  const torch::Tensor mps_random =
      torch::randint(255, {kTestBufferSize}, torch::TensorOptions().device(torch::kMPS).dtype(torch::kUInt8));

  std::memset(buffer.contents, 0, kTestBufferSize);
  torch::mps::synchronize();
  // Make sure we could interact with other mps tensors.
  mps_tensor += mps_random;
  torch::mps::synchronize();
  ASSERT_TENSORS_EQ(mps_tensor, mps_random);
  ASSERT_TENSORS_EQ_AND_PROPERTIES_UNCHANGED(cpu_tensor, mps_tensor);
}

#undef ASSERT_TENSORS_EQ_AND_PROPERTIES_UNCHANGED
#undef ASSERT_TENSORS_EQ
