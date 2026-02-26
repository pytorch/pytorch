#include <gtest/gtest.h>

#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Sleep.h>

TEST(CUDAEventTest, testCUDAExternalEvent) {
  if (!at::cuda::is_available()) {
    return;
  }

  // Create two external CUDA events
  unsigned int flags = cudaEventDefault | cudaEventExternal;
  auto event1 = at::cuda::CUDAEvent(flags);
  auto event2 = at::cuda::CUDAEvent(flags);
  // Ensure external CUDAEvent remain valid and functional after being moved.
  auto start_event = std::move(event1);
  auto end_event = std::move(event2);

  auto stream = at::cuda::getStreamFromPool();
  at::cuda::setCurrentCUDAStream(stream);

  auto graph = at::cuda::CUDAGraph();
  graph.capture_begin();
  start_event.record();
  at::cuda::sleep(100000);
  end_event.record();
  graph.capture_end();

  // External events should correctly record timestamps even when used inside
  // CUDA graphs, and elapsed_time() between them should be positive.
  stream.synchronize();
  graph.replay();
  at::cuda::device_synchronize();
  EXPECT_TRUE(start_event.elapsed_time(end_event) > 0);
}
