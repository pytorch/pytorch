#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSStream.h>

#include <atomic>
#include <chrono>
#include <thread>

namespace replay {
std::function<void()> callback_action;

class ReplayBufferCleaner : virtual public at::mps::IMpsAllocatorCallback {
    public:
    void executeMPSAllocatorCallback(void* ptr, EventType event) override {
     if (event == EventType::ALLOCATION_FAILED) {
        callback_action();
     }
    }
};
}

namespace at::mps {
REGISTER_MPS_ALLOCATOR_CALLBACK("ReplayBufferCleaner", replay::ReplayBufferCleaner);
}

TEST(MPSAllocator, MPSAllocatorCallbacks) {
    // fail if mps isn't available
    ASSERT_TRUE(torch::mps::is_available());

    std::vector<torch::Tensor> replay_buffer;
    replay::callback_action = [&]() {
        if (!replay_buffer.empty()) {
            replay_buffer.erase(replay_buffer.begin(), replay_buffer.begin() + (replay_buffer.size()/10));
        }
    };
    size_t max_iter = 100000;
    for (size_t i = 0; i < max_iter; i++) {
        torch::Tensor new_value = torch::randn({10000, 10000}, at::device(at::kMPS));
        // early stop the first time the callback is called
        if (replay_buffer.size() != i) {
            break;
        }
        replay_buffer.push_back(new_value);
    }
    // call synchronize() explicitly to wait for all MPS streams to
    // finish the Metal completionHandlers in MPSAllocator. Note that MPSAllocator
    // does this implicitly, but we call this for testing purposes.
    torch::mps::synchronize();
    ASSERT_TRUE(replay_buffer.size() < max_iter);
}

// Test that `MPSAllocator::waitForEvents` waits on all events that were created
// with `MPSAllocator::recordStream`, not just the most recent one.
//
// The strategy is to create and record two workloads on different streams that
// operate on the same input tensor, but force the second one to complete before
// the first one, and then check to make sure that `waitForEvents` still waits
// until after we commit the first one.
TEST(MPSAllocator, RecordStreamMultipleConsumers) {
    ASSERT_TRUE(torch::mps::is_available());

    auto* allocator = at::mps::getIMPSAllocator();
    at::mps::MPSStream* stream1 = at::mps::getStreamFromPool();
    at::mps::MPSStream* stream2 = at::mps::getStreamFromPool();
    torch::Tensor t = torch::zeros({500}, at::device(at::kMPS));
    torch::mps::synchronize();

    // Empty the cache to prevent any chance that this test falls into
    // `alloc_buffer_block`'s `release_cached_buffers` call later in the test,
    // which would commit and wait on all streams, silently completing `stream1`
    // before the test explicitly commits it.
    allocator->emptyCache();

    // Create workload and record on `stream1`, but don't commit it yet, to
    // prevent its completion until later.
    at::mps::setCurrentMPSStream(stream1);
    torch::Tensor tmp1 = t + 0;
    at::mps::setCurrentMPSStream(nullptr);
    t.record_stream(stream1->unwrap());

    // Create workload and record on `stream2`, then commit and wait for it.
    at::mps::setCurrentMPSStream(stream2);
    torch::Tensor tmp2 = t + 0;
    at::mps::setCurrentMPSStream(nullptr);
    t.record_stream(stream2->unwrap());
    stream2->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

    // Create a separate thread to start waiting for any outstanding events on
    // `t`'s pointer. Initially, it will be stuck waiting since we haven't
    // committed any work for `stream1` yet.
    std::atomic<bool> started_waiting{false};
    std::atomic<bool> finished_waiting{false};
    const void* ptr = t.data_ptr();
    std::thread waiter([&]() {
        started_waiting.store(true, std::memory_order_release);
        allocator->waitForEvents({ptr});
        finished_waiting.store(true, std::memory_order_release);
    });
    while (!started_waiting.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    // Even after waiting for some time, the `waiter` thread should not be
    // finished yet since we still haven't committed `stream1`.
    ASSERT_FALSE(finished_waiting.load(std::memory_order_acquire));
    // Now commit `stream1` so that it starts to run and the `waiter` thread can
    // finish since it was only waiting on the uncommitted `stream1`'s work.
    stream1->synchronize(at::mps::SyncType::COMMIT);
    waiter.join();
    // Double check that the `waiter` thread reported that it successfully
    // completed waiting on the events.
    ASSERT_TRUE(finished_waiting.load());
}
