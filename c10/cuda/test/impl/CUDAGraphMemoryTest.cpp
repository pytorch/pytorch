#include <c10/cuda/impl/CUDAGraphMemory.h>

#include <gtest/gtest.h>

#include <array>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace c10::cuda::CUDAGraphMemory {
namespace {

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

static_assert(
    !std::is_constructible_v<CaptureDAG, FakeCaptureDAGQuery&&>,
    "CaptureDAG must not retain a query temporary");
static_assert(
    !std::is_constructible_v<CaptureDAG, const FakeCaptureDAGQuery&&>,
    "CaptureDAG must not retain a const query temporary");

template <typename Handle>
Handle handle(char& storage) {
  return reinterpret_cast<Handle>(&storage);
}

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

} // namespace
} // namespace c10::cuda::CUDAGraphMemory
