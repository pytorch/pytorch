#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ATen/mps/MPSAllocatorInterface.h>

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

TEST(MPSAllocator, MPSAllocatorRegisterCPUBackedBuffer) {
    ASSERT_TRUE(torch::mps::is_available());

    torch::Tensor cpu_t = torch::randn({10000}, at::device(at::kCPU));

    auto* storage_impl = cpu_t.storage().unsafeGetStorageImpl();
    at::DataPtr& cpu_data_ptr = storage_impl->_mutable_data_ptr_no_checks();

    auto* mps_alloc = at::mps::getIMPSAllocator(true);

    at::DataPtr mps_data_ptr = mps_alloc->registerCPUBackedPtr(cpu_data_ptr.get(), storage_impl->nbytes());

    ASSERT_TRUE(mps_alloc->isSharedBuffer(mps_data_ptr.get()));

    const void* cpu_mapped = std::get<0>(mps_alloc->getSharedBufferPtr(mps_data_ptr.get()));

    ASSERT_EQ(cpu_data_ptr.get(), cpu_mapped);
}
