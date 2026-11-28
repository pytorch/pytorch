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

// ---------------------------------------------------------------------------
// IPC tests
// ---------------------------------------------------------------------------
#ifndef _WIN32

TEST_F(MemoryTest, IpcRoundTripSameProcess) {
  // Allocate, write a pattern, get an IPC handle, open it, verify the data
  // matches, then clean up.  All within a single process — fork-based tests
  // are in test_ipc.py.
  void* dev_ptr = nullptr;
  ASSERT_EQ(orMalloc(&dev_ptr, 16), orSuccess);

  const char src[16] = "ipc_test_data!!";
  ASSERT_EQ(orMemcpy(dev_ptr, src, 16, orMemcpyHostToDevice), orSuccess);

  char name[OR_IPC_HANDLE_MAX_LEN];
  ptrdiff_t offset = 0;
  ASSERT_EQ(
      orGetIpcMemHandle(dev_ptr, name, sizeof(name), &offset), orSuccess);
  EXPECT_EQ(offset, ptrdiff_t{0}); // dev_ptr is the allocation base
  EXPECT_GT(strlen(name), 0u);

  void* mapped = nullptr;
  size_t size = 0;
  ASSERT_EQ(orOpenIpcMemHandle(&mapped, name, &size), orSuccess);
  // orMalloc page-aligns: actual block size >= requested size.
  EXPECT_GE(size, 16u);
  EXPECT_EQ(memcmp(mapped, src, 16), 0);

  EXPECT_EQ(orCloseIpcMemHandle(mapped, size), orSuccess);
  EXPECT_EQ(orFree(dev_ptr), orSuccess);
}

TEST_F(MemoryTest, IpcHandleNullDevPtr) {
  // A null pointer is not a registered device allocation.
  char name[OR_IPC_HANDLE_MAX_LEN];
  ptrdiff_t offset = 0;
  EXPECT_EQ(
      orGetIpcMemHandle(nullptr, name, sizeof(name), &offset),
      orErrorUnknown);
}

TEST_F(MemoryTest, IpcHandleNameBufferTooSmall) {
  void* dev_ptr = nullptr;
  ASSERT_EQ(orMalloc(&dev_ptr, 8), orSuccess);

  char tiny[4]; // deliberately undersized
  ptrdiff_t offset = 0;
  EXPECT_EQ(
      orGetIpcMemHandle(dev_ptr, tiny, sizeof(tiny), &offset),
      orErrorUnknown);

  EXPECT_EQ(orFree(dev_ptr), orSuccess);
}

TEST_F(MemoryTest, IpcOpenNonExistentHandle) {
  // Opening a name that was never created must fail.
  void* mapped = nullptr;
  size_t size = 0;
  EXPECT_EQ(
      orOpenIpcMemHandle(&mapped, "/or_no_such_shm", &size),
      orErrorUnknown);
}

TEST_F(MemoryTest, IpcSequenceNumberProducesUniqueNames) {
  // Two consecutive handles for the same allocation must have different
  // names so a re-share never collides with the previous shm object.
  void* dev_ptr = nullptr;
  ASSERT_EQ(orMalloc(&dev_ptr, 8), orSuccess);

  char name1[OR_IPC_HANDLE_MAX_LEN];
  char name2[OR_IPC_HANDLE_MAX_LEN];
  ptrdiff_t off = 0;

  ASSERT_EQ(
      orGetIpcMemHandle(dev_ptr, name1, sizeof(name1), &off), orSuccess);
  void* m1 = nullptr;
  size_t sz1 = 0;
  ASSERT_EQ(orOpenIpcMemHandle(&m1, name1, &sz1), orSuccess); // also unlinks
  ASSERT_EQ(orCloseIpcMemHandle(m1, sz1), orSuccess);

  ASSERT_EQ(
      orGetIpcMemHandle(dev_ptr, name2, sizeof(name2), &off), orSuccess);
  EXPECT_STRNE(name1, name2);

  void* m2 = nullptr;
  size_t sz2 = 0;
  ASSERT_EQ(orOpenIpcMemHandle(&m2, name2, &sz2), orSuccess);
  ASSERT_EQ(orCloseIpcMemHandle(m2, sz2), orSuccess);

  EXPECT_EQ(orFree(dev_ptr), orSuccess);
}

TEST_F(MemoryTest, IpcOffsetForSuballocatedPointer) {
  // When ptr points into the interior of an allocation the returned offset
  // must equal the byte delta from the block base to ptr.
  void* dev_ptr = nullptr;
  ASSERT_EQ(orMalloc(&dev_ptr, 64), orSuccess);

  // Write a recognisable pattern at byte 16 via a full-block host copy.
  char host_src[64] = {};
  const char pattern[4] = {0x11, 0x22, 0x33, 0x44};
  memcpy(host_src + 16, pattern, 4);
  ASSERT_EQ(
      orMemcpy(dev_ptr, host_src, 64, orMemcpyHostToDevice), orSuccess);

  char* inner_ptr = static_cast<char*>(dev_ptr) + 16;
  char name[OR_IPC_HANDLE_MAX_LEN];
  ptrdiff_t offset = 0;
  ASSERT_EQ(
      orGetIpcMemHandle(inner_ptr, name, sizeof(name), &offset), orSuccess);
  EXPECT_EQ(offset, ptrdiff_t{16});

  void* mapped = nullptr;
  size_t size = 0;
  ASSERT_EQ(orOpenIpcMemHandle(&mapped, name, &size), orSuccess);
  // Apply the reported offset to reach the inner data.
  const char* consumer_ptr = static_cast<const char*>(mapped) + offset;
  EXPECT_EQ(memcmp(consumer_ptr, pattern, 4), 0);

  EXPECT_EQ(orCloseIpcMemHandle(mapped, size), orSuccess);
  EXPECT_EQ(orFree(dev_ptr), orSuccess);
}

#endif // !_WIN32

} // namespace
