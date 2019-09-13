#include <c10d/FileStore.hpp>
#include <c10d/ProcessGroupNCCL.hpp>
#include <c10d/test/CUDATest.hpp>
#include <c10d/test/TestUtils.hpp>
#include <gtest/gtest.h>
#include <torch/csrc/cuda/nccl.h>

using namespace c10d::test;

class WorkNCCLSimulateErrors : public c10d::ProcessGroupNCCL::WorkNCCL {
 public:
  WorkNCCLSimulateErrors(
      const std::vector<at::Device>& devices,
      bool simulate_error)
      : WorkNCCL(devices), simulate_error_(simulate_error) {}

  std::exception_ptr checkForNCCLErrors(
      const std::vector<std::shared_ptr<c10d::NCCLComm>>& ncclComms)
      const override {
    if (simulate_error_) {
      return std::make_exception_ptr(std::runtime_error("Error"));
    }
    return c10d::ProcessGroupNCCL::WorkNCCL::checkForNCCLErrors(ncclComms);
  }

 private:
  bool simulate_error_;
};

class ProcessGroupNCCLSimulateErrors : public c10d::ProcessGroupNCCL {
 public:
  ProcessGroupNCCLSimulateErrors(
      const std::shared_ptr<c10d::Store>& store,
      int rank,
      int size)
      : ProcessGroupNCCL(store, rank, size), simulate_error_(false) {}

  std::exception_ptr checkForNCCLErrors(
      const std::vector<std::shared_ptr<c10d::NCCLComm>>& ncclComms) override {
    if (simulate_error_) {
      return std::make_exception_ptr(std::runtime_error("Error"));
    }
    return c10d::ProcessGroupNCCL::checkForNCCLErrors(ncclComms);
  }

  std::chrono::duration<int64_t, std::milli> getWatchdogSleepInterval() {
    return std::chrono::milliseconds(
        ProcessGroupNCCLSimulateErrors::kWatchdogThreadSleepMillis);
  }

  std::shared_ptr<ProcessGroupNCCL::WorkNCCL> initWork(
      std::vector<at::Device> devices) override {
    return std::make_shared<WorkNCCLSimulateErrors>(devices, simulate_error_);
  }

  size_t getNCCLCommCacheSize() {
    return devNCCLCommMap_.size();
  }

  void simulate_error() {
    simulate_error_ = true;
  }

  void reset_error() {
    simulate_error_ = false;
  }

 private:
  bool simulate_error_;
};

class ProcessGroupNCCLErrorsTest : public ::testing::Test {
 protected:
  bool skipTest() {
    if (cudaNumDevices() == 0) {
      return true;
    }
#ifdef USE_C10D_NCCL
    return torch::cuda::nccl::version() < 2400;
#else
    return false;
#endif
  }

  void SetUp() override {
    size_t numDevices = cudaNumDevices();
    TemporaryFile file;
    store_ = std::make_shared<::c10d::FileStore>(file.path, 1);

    at::cuda::OptionalCUDAGuard deviceGuard;
    tensors_.resize(numDevices);
    for (auto i = 0; i < numDevices; ++i) {
      deviceGuard.set_index(i);
      tensors_[i] = at::ones({3, 3}, at::kCUDA);
    }
  }

  void TearDown() override {
    ASSERT_TRUE(setenv(c10d::NCCL_BLOCKING_WAIT, "0", 1) == 0);
  }

  std::vector<at::Tensor> tensors_;
  std::shared_ptr<::c10d::FileStore> store_;
};

TEST_F(ProcessGroupNCCLErrorsTest, testNCCLErrorsBlocking) {
  if (skipTest()) {
    return;
  }

  ASSERT_TRUE(setenv(c10d::NCCL_BLOCKING_WAIT, "1", 1) == 0);
  ProcessGroupNCCLSimulateErrors pg(store_, 0, 1);

  auto work = pg.allreduce(tensors_);
  work->wait();
  EXPECT_TRUE(work->isSuccess());
  EXPECT_EQ(1, pg.getNCCLCommCacheSize());

  // Now run all reduce with errors.
  pg.simulate_error();
  work = pg.allreduce(tensors_);
  EXPECT_THROW(work->wait(), std::runtime_error);

  // Verify the work item failed.
  EXPECT_TRUE(work->isCompleted());
  EXPECT_FALSE(work->isSuccess());
  EXPECT_THROW(work->wait(), std::runtime_error);

  // Should remove the nccl communicators which hit errors from the cache.
  std::this_thread::sleep_for(2 * pg.getWatchdogSleepInterval());
  EXPECT_EQ(0, pg.getNCCLCommCacheSize());

  // Verify we can recover from errors.
  pg.reset_error();
  work = pg.allreduce(tensors_);
  work->wait();
  EXPECT_TRUE(work->isSuccess());
  EXPECT_EQ(1, pg.getNCCLCommCacheSize());
}

TEST_F(ProcessGroupNCCLErrorsTest, testNCCLErrorsNonBlocking) {
  if (skipTest()) {
    return;
  }

  ProcessGroupNCCLSimulateErrors pg(store_, 0, 1);

  auto work = pg.allreduce(tensors_);
  pg.barrier()->wait();
  EXPECT_TRUE(work->isSuccess());
  EXPECT_EQ(1, pg.getNCCLCommCacheSize());

  // Now run all reduce with errors.
  pg.simulate_error();
  work = pg.allreduce(tensors_);

  // Should not throw exceptions.
  work->wait();
  pg.barrier()->wait();

  // Verify the work item failed.
  EXPECT_TRUE(work->isCompleted());
  EXPECT_FALSE(work->isSuccess());

  // Should remove the nccl communicators which hit errors from the cache.
  std::this_thread::sleep_for(2 * pg.getWatchdogSleepInterval());
  EXPECT_EQ(0, pg.getNCCLCommCacheSize());

  // Verify we can recover from errors.
  pg.reset_error();
  work = pg.allreduce(tensors_);
  pg.barrier()->wait();
  EXPECT_TRUE(work->isSuccess());
  EXPECT_EQ(1, pg.getNCCLCommCacheSize());
}
