#include <chrono>
#include <future>
#include <random>
#include <thread>
#include <array>

#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/core/hip/context_hip.h"
#include <gtest/gtest.h>

CAFFE2_DECLARE_bool(caffe2_hip_full_device_control);

namespace caffe2 {

namespace {
std::shared_ptr<void> shared_from_new(std::pair<void*, MemoryDeleter>&& p)
{
    return std::shared_ptr<void>(p.first, std::move(p.second));
}
}

TEST(HIPTest, HasHipRuntime) { EXPECT_TRUE(HasHipRuntime()); }

TEST(HIPContextTest, TestAllocDealloc)
{
    if(!HasHipGPU())
        return;
    HIPContext context(0);
    context.SwitchToDevice();
    auto data = shared_from_new(HIPContext::New(10 * sizeof(float)));
    EXPECT_NE(data.get(), nullptr);
}

TEST(HIPContextTest, TestSetGetDeviceWithoutCaffeMode)
{
    // For a while, set full device control to be true.
    for(int i = 0; i < NumHipDevices(); ++i)
    {
        CaffeHipSetDevice(i);
        EXPECT_EQ(CaffeHipGetDevice(), i);
    }
    for(int i = NumHipDevices() - 1; i >= 0; --i)
    {
        CaffeHipSetDevice(i);
        EXPECT_EQ(CaffeHipGetDevice(), i);
    }
}

TEST(HIPContextTest, TestSetGetDeviceWithCaffeMode)
{
    // For a while, set full device control to be true.
    FLAGS_caffe2_hip_full_device_control = true;
    for(int i = 0; i < NumHipDevices(); ++i)
    {
        CaffeHipSetDevice(i);
        EXPECT_EQ(CaffeHipGetDevice(), i);
    }
    for(int i = NumHipDevices() - 1; i >= 0; --i)
    {
        CaffeHipSetDevice(i);
        EXPECT_EQ(CaffeHipGetDevice(), i);
    }
    FLAGS_caffe2_hip_full_device_control = false;
}

TEST(HIPContextTest, MemoryPoolAllocateDealloc)
{
    if(!HasHipGPU())
        return;
    if(GetHipMemoryPoolType() == HipMemoryPoolType::NONE)
    {
        LOG(ERROR) << "Choose a memory type that is not none to test memory pool.";
        return;
    }
    const int nbytes = 1048576;
    for(int i = 0; i < NumHipDevices(); ++i)
    {
        LOG(INFO) << "Device " << i << " of " << NumHipDevices();
        DeviceGuard guard(i);
        auto allocated = shared_from_new(HIPContext::New(nbytes));
        EXPECT_NE(allocated, nullptr);
        hipPointerAttribute_t attr;
        HIP_ENFORCE(hipPointerGetAttributes(&attr, allocated.get()));
        EXPECT_EQ(attr.memoryType, hipMemoryTypeDevice);
        EXPECT_EQ(attr.device, i);
        void* prev_allocated = allocated.get();
        allocated.reset();
        auto new_allocated = shared_from_new(HIPContext::New(nbytes));
        // With a pool, the above allocation should yield the same address.
        EXPECT_EQ(new_allocated.get(), prev_allocated);
        // But, if we are allocating something larger, we will have a different
        // chunk of memory.
        auto larger_allocated = shared_from_new(HIPContext::New(nbytes * 2));
        EXPECT_NE(larger_allocated.get(), prev_allocated);
    }
}

hipStream_t getStreamForHandle(rocblas_handle handle)
{
    hipStream_t stream = nullptr;
    ROCBLAS_ENFORCE(rocblas_get_stream(handle, &stream));
    CHECK_NOTNULL(stream);
    return stream;
}

TEST(HIPContextTest, TestSameThreadSameObject)
{
    if(!HasHipGPU())
        return;
    HIPContext context_a(0);
    HIPContext context_b(0);
    EXPECT_EQ(context_a.hip_stream(), context_b.hip_stream());
    EXPECT_EQ(context_a.rocblas_handle(), context_b.rocblas_handle());
    EXPECT_EQ(
        context_a.hip_stream(), getStreamForHandle(context_b.rocblas_handle()));
    // hipRAND generators are context-local.
    EXPECT_NE(context_a.hiprand_generator(), context_b.hiprand_generator());
}

TEST(HIPContextTest, TestSameThreadDifferntObjectIfDifferentDevices)
{
    if(NumHipDevices() > 1)
    {
        HIPContext context_a(0);
        HIPContext context_b(1);
        EXPECT_NE(context_a.hip_stream(), context_b.hip_stream());
        EXPECT_NE(context_a.rocblas_handle(), context_b.rocblas_handle());
        EXPECT_NE(
            context_a.hip_stream(),
            getStreamForHandle(context_b.rocblas_handle()));
        EXPECT_NE(context_a.hiprand_generator(), context_b.hiprand_generator());
    }
}

namespace {
// A test function to return a stream address from a temp HIP context. You
// should not use that stream though, because the actual stream is destroyed
// after thread exit.
void TEST_GetStreamAddress(hipStream_t* ptr)
{
    HIPContext context(0);
    *ptr = context.hip_stream();
    // Sleep for a while so we have concurrent thread executions
    std::this_thread::sleep_for(std::chrono::seconds(1));
}
} // namespace

TEST(HIPContextTest, TestDifferntThreadDifferentobject)
{
    if(!HasHipGPU())
        return;
    std::array<hipStream_t, 2> temp = {0};
    // Same thread
    TEST_GetStreamAddress(&temp[0]);
    TEST_GetStreamAddress(&temp[1]);
    EXPECT_TRUE(temp[0] != nullptr);
    EXPECT_TRUE(temp[1] != nullptr);
    EXPECT_EQ(temp[0], temp[1]);
    // Different threads
    std::thread thread_a(TEST_GetStreamAddress, &temp[0]);
    std::thread thread_b(TEST_GetStreamAddress, &temp[1]);
    thread_a.join();
    thread_b.join();
    EXPECT_TRUE(temp[0] != nullptr);
    EXPECT_TRUE(temp[1] != nullptr);
    EXPECT_NE(temp[0], temp[1]);
}

} // namespace caffe2
