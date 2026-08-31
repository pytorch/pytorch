#include <c10/cuda/impl/CUDAGraphMemory.h>

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace c10::cuda::CUDAGraphMemory {
namespace {

template <typename Handle>
Handle handle(char& storage) {
  return reinterpret_cast<Handle>(&storage);
}

c10::cuda::CUDAStream testStream(
    cudaStream_t stream,
    c10::DeviceIndex device_index = 0) {
  return c10::cuda::getStreamFromExternal(stream, device_index);
}

class NoopCaptureDAGQuery final : public CaptureDAGQuery {
 public:
  CaptureDAGInfo captureInfo(cudaStream_t) const override {
    return {};
  }

  std::vector<cudaGraphNode_t> dependencies(cudaGraphNode_t) const override {
    return {};
  }
};

struct TestPool {
  c10::MempoolId_t id;

  c10::MempoolId_t owner_MempoolId() const {
    return id;
  }
};

struct TestBlock {
  cudaStream_t stream;
  ska::flat_hash_set<c10::cuda::CUDAStream> stream_uses;
  TestPool* pool;
};

struct TestAllocatorOps {
  std::vector<TestBlock*> freed_blocks;
  std::vector<TestBlock*> event_blocks;
  std::vector<ska::flat_hash_set<c10::cuda::CUDAStream>> event_streams;
  bool fail_insert_events{false};

  c10::MempoolId_t capturePoolForStream(cudaStream_t) {
    return {1, 0};
  }

  void freeBlock(TestBlock* block) {
    freed_blocks.push_back(block);
  }

  void insertEvents(TestBlock* block) {
    if (fail_insert_events) {
      block->stream_uses.clear();
      TORCH_CHECK(false, "injected allocator event insertion failure");
    }
    event_blocks.push_back(block);
    event_streams.push_back(block->stream_uses);
    block->stream_uses.clear();
  }
};

constexpr auto kDefer = DeferredFreePolicy::DEFER_UNTIL_NO_ACTIVE_CAPTURE;

TEST(CUDAGraphMemoryBlockManagerTest, PreservesPreCaptureDeviceQualifiedUse) {
  char stream_storage{};
  auto raw_stream = handle<cudaStream_t>(stream_storage);
  auto pre_capture_stream = testStream(raw_stream, 0);
  auto capture_stream = testStream(raw_stream, 1);
  NoopCaptureDAGQuery query;
  BlockManager<TestBlock> manager(query);
  TestPool pool{{1, 0}};
  TestBlock block{raw_stream, {pre_capture_stream, capture_stream}, &pool};

  manager.recordStreamUse(&block, capture_stream, 1);
  manager.deferFree(&block, kDefer);
  TestAllocatorOps ops;
  manager.drainDeferredBlocks(ops);

  ASSERT_EQ(ops.event_blocks.size(), 1);
  EXPECT_EQ(ops.event_blocks.front(), &block);
  ASSERT_EQ(ops.event_streams.size(), 1);
  EXPECT_EQ(
      ops.event_streams.front(),
      ska::flat_hash_set<c10::cuda::CUDAStream>{pre_capture_stream});
  EXPECT_FALSE(manager.contains(&block));
}

TEST(CUDAGraphMemoryBlockManagerTest, RestoresStreamUsesAfterEventFailure) {
  std::array<char, 3> stream_storage{};
  auto first = testStream(handle<cudaStream_t>(stream_storage[0]));
  auto second = testStream(handle<cudaStream_t>(stream_storage[1]));
  auto capture = testStream(handle<cudaStream_t>(stream_storage[2]));
  NoopCaptureDAGQuery query;
  BlockManager<TestBlock> manager(query);
  TestPool pool{{1, 0}};
  TestBlock block{capture.stream(), {first, second, capture}, &pool};
  manager.recordStreamUse(&block, capture, 1);
  manager.deferFree(&block, kDefer);

  const ska::flat_hash_set<c10::cuda::CUDAStream> expected{first, second};
  TestAllocatorOps ops;
  ops.fail_insert_events = true;
  EXPECT_THROW(manager.drainDeferredBlocks(ops), c10::Error);
  EXPECT_EQ(block.stream_uses, expected);
  EXPECT_TRUE(manager.isDeferred(&block));

  ops.fail_insert_events = false;
  manager.drainDeferredBlocks(ops);
  ASSERT_EQ(ops.event_blocks.size(), 1);
  EXPECT_EQ(ops.event_streams.front(), expected);
  EXPECT_FALSE(manager.contains(&block));
}

} // namespace
} // namespace c10::cuda::CUDAGraphMemory
