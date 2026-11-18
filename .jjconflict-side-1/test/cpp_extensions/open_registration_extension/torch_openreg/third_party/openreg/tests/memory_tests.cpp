#include <gtest/gtest.h>
#include <include/openreg.h>

namespace {

class MemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    orSetDevice(0);
  }
};

TEST_F(MemoryTest, AllocateAndFreeDevice) {
  void* ptr = nullptr;
  EXPECT_EQ(orMalloc(&ptr, 4096), orSuccess);
  EXPECT_NE(ptr, nullptr);

  EXPECT_EQ(orFree(ptr), orSuccess);
}

TEST_F(MemoryTest, AllocateAndFreeHost) {
  void* ptr = nullptr;
  EXPECT_EQ(orMallocHost(&ptr, 8192), orSuccess);
  EXPECT_NE(ptr, nullptr);

  EXPECT_EQ(orFreeHost(ptr), orSuccess);
}

TEST_F(MemoryTest, FreeNullptrIsNoop) {
  // Freeing a nullptr should behave like CUDA: treated as a no-op success.
  EXPECT_EQ(orFree(nullptr), orSuccess);
  EXPECT_EQ(orFreeHost(nullptr), orSuccess);
}

TEST_F(MemoryTest, AllocateNullptr) {
  EXPECT_EQ(orMalloc(nullptr, 4096), orErrorUnknown);
  EXPECT_EQ(orMallocHost(nullptr, 4096), orErrorUnknown);
}

TEST_F(MemoryTest, AllocateZeroSize) {
  void* ptr = nullptr;
  EXPECT_EQ(orMalloc(&ptr, 0), orErrorUnknown);
  EXPECT_EQ(orMallocHost(&ptr, 0), orErrorUnknown);
}

TEST_F(MemoryTest, MemcpyHostToDevice) {
  char host_src[] = "data";
  char host_dst[5] = {};

  void* dev_ptr = nullptr;
  EXPECT_EQ(orMalloc(&dev_ptr, 5), orSuccess);

  EXPECT_EQ(orMemcpy(dev_ptr, host_src, 5, orMemcpyHostToDevice), orSuccess);
  EXPECT_EQ(orMemcpy(host_dst, dev_ptr, 5, orMemcpyDeviceToHost), orSuccess);

  EXPECT_STREQ(host_dst, host_src);

  EXPECT_EQ(orFree(dev_ptr), orSuccess);
}

TEST_F(MemoryTest, MemcpyDeviceToDevice) {
  const char host_src[5] = "data";
  char host_dst[5] = {};
  void *dev_dst1 = nullptr, *dev_dst2 = nullptr;

  EXPECT_EQ(orMalloc(&dev_dst1, 5), orSuccess);
  EXPECT_EQ(orMalloc(&dev_dst2, 5), orSuccess);

  EXPECT_EQ(orMemcpy(dev_dst1, host_src, 5, orMemcpyHostToDevice), orSuccess);
  EXPECT_EQ(orMemcpy(dev_dst2, dev_dst1, 5, orMemcpyDeviceToDevice), orSuccess);
  EXPECT_EQ(orMemcpy(host_dst, dev_dst2, 5, orMemcpyDeviceToHost), orSuccess);

  EXPECT_STREQ(host_dst, host_src);

  EXPECT_EQ(orFree(dev_dst1), orSuccess);
  EXPECT_EQ(orFree(dev_dst2), orSuccess);
}

TEST_F(MemoryTest, MemcpyInvalidKind) {
  char host_ptr[5] = "data";
  void* dev_ptr = nullptr;

  EXPECT_EQ(orMalloc(&dev_ptr, 5), orSuccess);

  EXPECT_EQ(
      orMemcpy(nullptr, host_ptr, 4, orMemcpyHostToDevice), orErrorUnknown);
  EXPECT_EQ(
      orMemcpy(dev_ptr, nullptr, 4, orMemcpyHostToDevice), orErrorUnknown);
  EXPECT_EQ(
      orMemcpy(dev_ptr, host_ptr, 0, orMemcpyHostToDevice), orErrorUnknown);

  EXPECT_EQ(orFree(dev_ptr), orSuccess);
}

TEST_F(MemoryTest, MemcpyInvalidCombinations) {
  void *dev_src = nullptr, *dev_dst = nullptr;
  EXPECT_EQ(orMalloc(&dev_src, 8), orSuccess);
  EXPECT_EQ(orMalloc(&dev_dst, 8), orSuccess);

  char host_buf[8] = {};

  // Deliberately pass mismatched kinds to ensure validation coverage.
  EXPECT_EQ(
      orMemcpy(host_buf, dev_src, 4, orMemcpyHostToDevice), orErrorUnknown);
  EXPECT_EQ(
      orMemcpy(dev_dst, host_buf, 4, orMemcpyDeviceToHost), orErrorUnknown);
  EXPECT_EQ(
      orMemcpy(dev_dst, dev_src, 4, orMemcpyHostToDevice), orErrorUnknown);

  EXPECT_EQ(orFree(dev_src), orSuccess);
  EXPECT_EQ(orFree(dev_dst), orSuccess);
}

TEST_F(MemoryTest, MemcpyAsyncHostToDevice) {
  orStream_t stream = nullptr;
  EXPECT_EQ(orStreamCreate(&stream), orSuccess);

  const char host_src[] = "async";
  char host_dst[6] = {};
  void* dev_ptr = nullptr;
  EXPECT_EQ(orMalloc(&dev_ptr, sizeof(host_src)), orSuccess);

  // Async copies should complete once the stream is synchronized.
  EXPECT_EQ(
      orMemcpyAsync(dev_ptr, host_src, sizeof(host_src), orMemcpyHostToDevice, stream),
      orSuccess);
  EXPECT_EQ(orStreamSynchronize(stream), orSuccess);
  EXPECT_EQ(orMemcpy(
                host_dst, dev_ptr, sizeof(host_src), orMemcpyDeviceToHost),
            orSuccess);
  EXPECT_STREQ(host_dst, host_src);

  EXPECT_EQ(orFree(dev_ptr), orSuccess);
  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

TEST_F(MemoryTest, PointerAttributes) {
  void* dev_ptr = nullptr;
  EXPECT_EQ(orMalloc(&dev_ptr, 32), orSuccess);

  orPointerAttributes attr{};
  EXPECT_EQ(orPointerGetAttributes(&attr, dev_ptr), orSuccess);
  EXPECT_EQ(attr.type, orMemoryType::orMemoryTypeDevice);
  EXPECT_EQ(attr.pointer, dev_ptr);

  char host_ptr[16];
  EXPECT_EQ(orPointerGetAttributes(&attr, host_ptr), orSuccess);
  EXPECT_EQ(attr.type, orMemoryType::orMemoryTypeUnmanaged);

  EXPECT_EQ(orFree(dev_ptr), orSuccess);
}

TEST_F(MemoryTest, PointerAttributesInvalidArgs) {
  // Attribute queries must fail on null inputs to avoid dereferencing.
  char buffer[8] = {};
  orPointerAttributes attr{};
  EXPECT_EQ(orPointerGetAttributes(nullptr, buffer), orErrorUnknown);
  EXPECT_EQ(orPointerGetAttributes(&attr, nullptr), orErrorUnknown);
}

TEST_F(MemoryTest, ProtectUnprotectDevice) {
  void* dev_ptr = nullptr;
  EXPECT_EQ(orMalloc(&dev_ptr, 64), orSuccess);

  EXPECT_EQ(orMemoryUnprotect(dev_ptr), orSuccess);
  EXPECT_EQ(orMemoryProtect(dev_ptr), orSuccess);

  EXPECT_EQ(orFree(dev_ptr), orSuccess);
}

TEST_F(MemoryTest, ProtectReferenceCounting) {
  void* dev_ptr = nullptr;
  EXPECT_EQ(orMalloc(&dev_ptr, 64), orSuccess);

  // Call unprotect/protect twice to exercise the refcount transitions.
  EXPECT_EQ(orMemoryUnprotect(dev_ptr), orSuccess);
  EXPECT_EQ(orMemoryUnprotect(dev_ptr), orSuccess);
  EXPECT_EQ(orMemoryProtect(dev_ptr), orSuccess);
  EXPECT_EQ(orMemoryProtect(dev_ptr), orSuccess);

  EXPECT_EQ(orFree(dev_ptr), orSuccess);
}

TEST_F(MemoryTest, DoubleFreeFails) {
  void* dev_ptr = nullptr;
  EXPECT_EQ(orMalloc(&dev_ptr, 32), orSuccess);
  EXPECT_EQ(orFree(dev_ptr), orSuccess);
  EXPECT_EQ(orFree(dev_ptr), orErrorUnknown);
}

} // namespace
