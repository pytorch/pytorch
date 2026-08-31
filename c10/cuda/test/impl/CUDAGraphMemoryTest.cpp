#include <c10/cuda/impl/CUDAGraphMemory.h>

#include <gtest/gtest.h>

#include <array>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

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

TEST(CUDAGraphMemoryCaptureTrackerTest, TracksConditionalCaptureTree) {
  CaptureTracker capture_tracker;

  capture_tracker.captureBegin(
      {1, {1, 1}, nullptr, std::nullopt, std::nullopt});
  capture_tracker.captureBegin(
      {2, {1, 1}, nullptr, 1, std::optional<cudaStream_t>{nullptr}});
  EXPECT_TRUE(capture_tracker.hasActiveCaptures());

  capture_tracker.captureEnd(2);
  EXPECT_TRUE(capture_tracker.hasActiveCaptures());
  capture_tracker.captureEnd(1);
  EXPECT_FALSE(capture_tracker.hasActiveCaptures());
}

TEST(CUDAGraphMemoryCaptureTrackerTest, RootEndClearsActiveDescendants) {
  CaptureTracker capture_tracker;

  capture_tracker.captureBegin(
      {1, {1, 1}, nullptr, std::nullopt, std::nullopt});
  capture_tracker.captureBegin(
      {2, {1, 1}, nullptr, 1, std::optional<cudaStream_t>{nullptr}});
  capture_tracker.captureEnd(1);

  EXPECT_FALSE(capture_tracker.hasActiveCaptures());
  EXPECT_THROW(capture_tracker.captureEnd(2), c10::Error);
}

class FakeCaptureDAGQuery final : public CaptureDAGQuery {
 public:
  CaptureDAGInfo captureInfo(cudaStream_t stream) const override {
    ++num_capture_queries;
    return captures.at(stream);
  }

  std::vector<cudaGraphNode_t> dependencies(
      cudaGraphNode_t node) const override {
    ++num_dependency_queries;
    auto it = parents.find(node);
    return it == parents.end() ? std::vector<cudaGraphNode_t>{} : it->second;
  }

  std::unordered_map<cudaStream_t, CaptureDAGInfo> captures;
  std::unordered_map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> parents;
  mutable size_t num_capture_queries{0};
  mutable size_t num_dependency_queries{0};
};

static_assert(!std::is_constructible_v<CaptureDAG, FakeCaptureDAGQuery&&>);

TEST(CaptureDAGTest, AggregatesOnlyMarkersFromOneActiveCapture) {
  std::array<char, 5> stream_storage{};
  std::array<char, 2> graph_storage{};
  std::array<char, 3> node_storage{};
  auto stream1 = handle<cudaStream_t>(stream_storage[0]);
  auto stream2 = handle<cudaStream_t>(stream_storage[1]);
  auto noncapturing = handle<cudaStream_t>(stream_storage[2]);
  auto other_capture = handle<cudaStream_t>(stream_storage[3]);
  auto invalidated = handle<cudaStream_t>(stream_storage[4]);
  auto graph = handle<cudaGraph_t>(graph_storage[0]);
  std::array<cudaGraphNode_t, 2> terminals1 = {
      handle<cudaGraphNode_t>(node_storage[0]),
      handle<cudaGraphNode_t>(node_storage[1])};
  std::array<cudaGraphNode_t, 2> terminals2 = {
      handle<cudaGraphNode_t>(node_storage[1]),
      handle<cudaGraphNode_t>(node_storage[2])};

  FakeCaptureDAGQuery query;
  query.captures[stream1] = {
      graph,
      1,
      terminals1.data(),
      terminals1.size(),
      cudaStreamCaptureStatusActive};
  query.captures[stream2] = {
      graph,
      1,
      terminals2.data(),
      terminals2.size(),
      cudaStreamCaptureStatusActive};
  query.captures[noncapturing] = {
      nullptr, 0, nullptr, 0, cudaStreamCaptureStatusNone};
  query.captures[other_capture] = {
      handle<cudaGraph_t>(graph_storage[1]),
      2,
      nullptr,
      0,
      cudaStreamCaptureStatusActive};
  query.captures[invalidated] = {
      nullptr, 3, nullptr, 0, cudaStreamCaptureStatusInvalidated};

  CaptureDAG dag(query);
  CaptureDAG::FreeMarkerState markers;
  EXPECT_TRUE(dag.recordFreeMarkersForStream(stream1, markers));
  EXPECT_TRUE(dag.recordFreeMarkersForStream(stream2, markers));
  auto unique_markers = dag.takeFreeMarkers(std::move(markers));
  EXPECT_EQ(
      std::unordered_set<cudaGraphNode_t>(
          unique_markers.begin(), unique_markers.end())
          .size(),
      3);

  CaptureDAG::FreeMarkerState inactive;
  EXPECT_FALSE(dag.recordFreeMarkersForStream(noncapturing, inactive));
  CaptureDAG::FreeMarkerState invalid;
  EXPECT_FALSE(dag.recordFreeMarkersForStream(invalidated, invalid));
  CaptureDAG::FreeMarkerState mixed;
  EXPECT_TRUE(dag.recordFreeMarkersForStream(stream1, mixed));
  EXPECT_FALSE(dag.recordFreeMarkersForStream(other_capture, mixed));
}

TEST(CaptureDAGTest, TraversesDependenciesIncrementally) {
  std::array<char, 5> node_storage{};
  auto root = handle<cudaGraphNode_t>(node_storage[0]);
  auto left = handle<cudaGraphNode_t>(node_storage[1]);
  auto right = handle<cudaGraphNode_t>(node_storage[2]);
  auto terminal1 = handle<cudaGraphNode_t>(node_storage[3]);
  auto terminal2 = handle<cudaGraphNode_t>(node_storage[4]);
  std::array<cudaGraphNode_t, 2> terminals = {terminal1, terminal2};

  FakeCaptureDAGQuery query;
  query.parents[terminal1] = {left};
  query.parents[terminal2] = {right};
  query.parents[left] = {root};
  query.parents[right] = {root};
  CaptureDAG dag(query);
  CaptureDAGInfo info{
      nullptr,
      1,
      terminals.data(),
      terminals.size(),
      cudaStreamCaptureStatusActive};
  CaptureDAG::TraversalState state;

  dag.updateVisited(info, state);
  const std::array<cudaGraphNode_t, 5> reachable = {
      root, left, right, terminal1, terminal2};
  EXPECT_TRUE(dag.areMarkersReachable(reachable, state));
  EXPECT_EQ(query.num_dependency_queries, 5);

  dag.updateVisited(info, state);
  EXPECT_EQ(query.num_dependency_queries, 5);

  char missing_storage{};
  const std::array<cudaGraphNode_t, 1> missing = {
      handle<cudaGraphNode_t>(missing_storage)};
  EXPECT_FALSE(dag.areMarkersReachable(missing, state));
  CaptureDAGInfo other_capture{
      handle<cudaGraph_t>(missing_storage),
      2,
      nullptr,
      0,
      cudaStreamCaptureStatusActive};
  EXPECT_THROW(dag.updateVisited(other_capture, state), c10::Error);
}

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
  c10::MempoolId_t capture_pool{1, 0};
  size_t pool_resolutions{0};
  std::vector<TestBlock*> freed_blocks;
  std::vector<TestBlock*> event_blocks;
  std::vector<ska::flat_hash_set<c10::cuda::CUDAStream>> event_streams;
  bool fail_insert_events{false};

  c10::MempoolId_t capturePoolForStream(cudaStream_t) {
    ++pool_resolutions;
    return capture_pool;
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

constexpr auto kReuse = DeferredFreePolicy::REUSE_WHEN_TOPOLOGICALLY_SAFE;
constexpr auto kDefer = DeferredFreePolicy::DEFER_UNTIL_NO_ACTIVE_CAPTURE;

TEST(CUDAGraphMemoryBlockManagerTest, ReclaimsTopologicallySafeBlock) {
  std::array<char, 2> stream_storage{};
  char graph_storage{};
  std::array<char, 3> node_storage{};
  auto allocation_stream = handle<cudaStream_t>(stream_storage[0]);
  auto use_stream = handle<cudaStream_t>(stream_storage[1]);
  auto graph = handle<cudaGraph_t>(graph_storage);
  auto allocation_marker = handle<cudaGraphNode_t>(node_storage[0]);
  auto use_marker = handle<cudaGraphNode_t>(node_storage[1]);
  auto frontier = handle<cudaGraphNode_t>(node_storage[2]);
  FakeCaptureDAGQuery query;
  query.captures[allocation_stream] = {
      graph, 1, &allocation_marker, 1, cudaStreamCaptureStatusActive};
  query.captures[use_stream] = {
      graph, 1, &use_marker, 1, cudaStreamCaptureStatusActive};

  BlockManager<TestBlock> manager(query);
  TestPool pool{{1, 0}};
  TestBlock block{allocation_stream, {testStream(use_stream)}, &pool};
  manager.recordAllocation(
      &block, {true, 1, allocation_stream, allocation_stream});
  manager.recordStreamUse(&block, testStream(use_stream), 1);
  manager.recordFree(&block, allocation_stream);
  manager.deferFree(&block, kReuse);

  query.parents[frontier] = {allocation_marker, use_marker};
  query.captures[allocation_stream].terminals = &frontier;
  TestAllocatorOps ops;
  manager.reclaimBlocks(allocation_stream, kReuse, ops);

  ASSERT_EQ(ops.freed_blocks.size(), 1);
  EXPECT_EQ(ops.freed_blocks.front(), &block);
  EXPECT_EQ(ops.pool_resolutions, 1);
  EXPECT_TRUE(block.stream_uses.empty());
  EXPECT_FALSE(manager.contains(&block));
}

TEST(CUDAGraphMemoryBlockManagerTest, EndsOnlyTheRequestedPool) {
  char stream_storage{};
  auto stream = handle<cudaStream_t>(stream_storage);
  FakeCaptureDAGQuery query;
  query.captures[stream] = {
      nullptr, 0, nullptr, 0, cudaStreamCaptureStatusNone};
  BlockManager<TestBlock> manager(query);
  TestPool first_pool{{1, 0}};
  TestPool second_pool{{2, 0}};
  TestBlock first{stream, {testStream(stream)}, &first_pool};
  TestBlock second{stream, {testStream(stream)}, &second_pool};

  manager.recordAllocation(&first, {true, 1, stream, stream});
  manager.recordStreamUse(&first, testStream(stream), 1);
  manager.recordFree(&first, stream);
  manager.deferFree(&first, kReuse);
  manager.recordAllocation(&second, {true, 2, stream, stream});
  manager.recordStreamUse(&second, testStream(stream), 2);
  manager.recordFree(&second, stream);
  manager.deferFree(&second, kReuse);

  TestAllocatorOps ops;
  manager.endCapturePool({1, 0}, kReuse, ops);
  ASSERT_EQ(ops.freed_blocks.size(), 1);
  EXPECT_EQ(ops.freed_blocks.front(), &first);
  EXPECT_FALSE(manager.contains(&first));
  EXPECT_TRUE(manager.contains(&second));
}

TEST(CUDAGraphMemoryBlockManagerTest, PreservesPreCaptureDeviceQualifiedUse) {
  char stream_storage{};
  auto raw_stream = handle<cudaStream_t>(stream_storage);
  auto pre_capture_stream = testStream(raw_stream, 0);
  auto capture_stream = testStream(raw_stream, 1);
  FakeCaptureDAGQuery query;
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

TEST(
    CUDAGraphMemoryBlockManagerTest,
    WaitsForAllCapturesBeforeRecordingUnknownStreamUse) {
  char stream_storage{};
  auto stream = handle<cudaStream_t>(stream_storage);
  FakeCaptureDAGQuery query;
  BlockManager<TestBlock> manager(query);
  TestPool pool{{1, 0}};
  TestBlock block{stream, {testStream(stream)}, &pool};

  manager.recordStreamUse(&block, testStream(stream), std::nullopt);
  manager.deferFree(&block, kDefer);
  manager.captureBegin();

  TestAllocatorOps ops;
  manager.drainDeferredBlocks(ops);
  EXPECT_TRUE(ops.freed_blocks.empty());
  EXPECT_TRUE(manager.contains(&block));

  manager.captureEnd();
  manager.drainDeferredBlocks(ops);
  EXPECT_TRUE(ops.freed_blocks.empty());
  ASSERT_EQ(ops.event_blocks.size(), 1);
  EXPECT_EQ(ops.event_blocks.front(), &block);
  EXPECT_EQ(
      ops.event_streams.front(),
      ska::flat_hash_set<c10::cuda::CUDAStream>{testStream(stream)});
  EXPECT_FALSE(manager.contains(&block));
}

TEST(CUDAGraphMemoryBlockManagerTest, RestoresStreamUsesAfterEventFailure) {
  std::array<char, 3> stream_storage{};
  auto first = testStream(handle<cudaStream_t>(stream_storage[0]));
  auto second = testStream(handle<cudaStream_t>(stream_storage[1]));
  auto capture = testStream(handle<cudaStream_t>(stream_storage[2]));
  FakeCaptureDAGQuery query;
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

TEST(CUDAGraphMemoryBlockManagerTest, DefersStreamUseOwnedByAnotherCapture) {
  alignas(8) std::array<char, 16> stream_storage{};
  std::array<char, 2> graph_storage{};
  std::array<char, 2> node_storage{};
  auto allocation_stream = handle<cudaStream_t>(stream_storage[0]);
  auto use_stream = handle<cudaStream_t>(stream_storage[8]);
  auto allocation_graph = handle<cudaGraph_t>(graph_storage[0]);
  auto use_graph = handle<cudaGraph_t>(graph_storage[1]);
  auto allocation_marker = handle<cudaGraphNode_t>(node_storage[0]);
  auto use_marker = handle<cudaGraphNode_t>(node_storage[1]);

  FakeCaptureDAGQuery query;
  query.captures[allocation_stream] = {
      allocation_graph,
      1,
      &allocation_marker,
      1,
      cudaStreamCaptureStatusActive};
  query.captures[use_stream] = {
      use_graph, 2, &use_marker, 1, cudaStreamCaptureStatusActive};
  BlockManager<TestBlock> manager(query);
  manager.captureBegin(
      {1, {1, 0}, allocation_stream, std::nullopt, std::nullopt});
  manager.captureBegin({2, {2, 0}, use_stream, std::nullopt, std::nullopt});
  TestPool pool{{1, 0}};
  TestBlock block{allocation_stream, {testStream(use_stream)}, &pool};
  manager.recordAllocation(
      &block, {true, 1, allocation_stream, allocation_stream});
  manager.recordStreamUse(&block, testStream(use_stream), 2);
  manager.recordFree(&block, allocation_stream);
  manager.deferFree(&block, kReuse);

  query.captures[allocation_stream].status = cudaStreamCaptureStatusNone;
  manager.captureEnd(1);
  TestAllocatorOps ops;
  manager.endCapturePool({1, 0}, kReuse, ops);
  EXPECT_TRUE(ops.freed_blocks.empty());
  EXPECT_TRUE(ops.event_blocks.empty());
  EXPECT_TRUE(manager.isDeferred(&block));

  query.captures[use_stream].status = cudaStreamCaptureStatusNone;
  manager.captureEnd(2);
  manager.drainDeferredBlocks(ops);
  ASSERT_EQ(ops.freed_blocks.size(), 1);
  EXPECT_EQ(ops.freed_blocks.front(), &block);
  EXPECT_FALSE(manager.contains(&block));
}

TEST(CUDAGraphMemoryBlockManagerTest, DoesNotReclaimInvalidatedCapture) {
  char stream_storage{};
  char graph_storage{};
  char node_storage{};
  auto stream = handle<cudaStream_t>(stream_storage);
  auto graph = handle<cudaGraph_t>(graph_storage);
  auto marker = handle<cudaGraphNode_t>(node_storage);
  FakeCaptureDAGQuery query;
  query.captures[stream] = {
      graph, 1, &marker, 1, cudaStreamCaptureStatusActive};
  BlockManager<TestBlock> manager(query);
  TestPool pool{{1, 0}};
  TestBlock block{stream, {}, &pool};
  manager.recordAllocation(&block, {true, 1, stream, stream});
  manager.recordFree(&block, stream);
  manager.deferFree(&block, kReuse);

  query.captures[stream].status = cudaStreamCaptureStatusInvalidated;
  TestAllocatorOps ops;
  manager.reclaimBlocks(stream, kReuse, ops);

  EXPECT_EQ(ops.pool_resolutions, 0);
  EXPECT_EQ(query.num_dependency_queries, 0);
  EXPECT_TRUE(ops.freed_blocks.empty());
  EXPECT_TRUE(manager.isDeferred(&block));
}

} // namespace
} // namespace c10::cuda::CUDAGraphMemory
