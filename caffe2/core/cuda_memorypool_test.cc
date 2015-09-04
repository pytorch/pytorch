#include "caffe2/core/cuda_memorypool.h"
#include "caffe2/core/context_gpu.h"
#include "gtest/gtest.h"
#include "glog/logging.h"

namespace caffe2 {

struct UseMemoryPool { static const bool value = true; };
struct NotUseMemoryPool { static const bool value = false; };

template <class UsePoolOrNot>
class MemoryPoolTest : public ::testing::Test {
 protected:
  MemoryPoolTest() : device_count_(0) {}
  // virtual void SetUp() will be called before each test is run.  You
  // should define it if you need to initialize the varaibles.
  // Otherwise, this can be skipped.
  void SetUp() override {
    int device_count_;
    CUDA_CHECK(cudaGetDeviceCount(&device_count_));
    // If we test with the memory pool, initialize the memory pool.
    if (UsePoolOrNot::value) {
      vector<int> device_ids(device_count_);
      for (int i = 0; i < device_count_; ++i) {
        device_ids[i] = i;
      }
      CHECK(CudaMemoryPool::InitializeMemoryPool(device_ids, 0.8));
    }
  }

  void TearDown() override {
    if (UsePoolOrNot::value) {
      CHECK(CudaMemoryPool::FinalizeMemoryPool());
    }
  }

  // Declares the variables your tests want to use.
  int device_count_;
};

typedef ::testing::Types<UseMemoryPool, NotUseMemoryPool> MemoryPoolTestTypes;
TYPED_TEST_CASE(MemoryPoolTest, MemoryPoolTestTypes);

// This just tests that setup and teardown works.
TYPED_TEST(MemoryPoolTest, InitializeAndFinalizeWorks) {
  EXPECT_TRUE(true);
}

TYPED_TEST(MemoryPoolTest, AllocateAndDeallocate) {
  const int nbytes = 1048576;
  for (int i = 0; i < this->device_count_; ++i) {
    LOG(INFO) << "Device " << i << " of " << this->device_count_;
    CUDA_CHECK(cudaSetDevice(i));
    void* allocated = CUDAContext::New(nbytes);
    EXPECT_NE(allocated, nullptr);
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, allocated));
    EXPECT_EQ(attr.memoryType, cudaMemoryTypeDevice);
    EXPECT_EQ(attr.device, i);
    CUDAContext::Delete(allocated);
  }
}

}  // namespace caffe2
