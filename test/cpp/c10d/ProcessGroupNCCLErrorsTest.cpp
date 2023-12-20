#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include <c10/util/irange.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include "CUDATest.hpp"
#include "TestUtils.hpp"

#include <gtest/gtest.h>

using namespace c10d::test;

constexpr int kNcclErrorHandlingVersion = 2400;

class WorkNCCLSimulateErrors : public c10d::ProcessGroupNCCL::WorkNCCL {
 public:
  WorkNCCLSimulateErrors(
      const std::vector<at::Device>& devices,
      bool simulate_error,
      int rank,
      c10d::OpType opType,
      uint64_t seq)
      : WorkNCCL(devices, rank, opType, seq), simulateError_(simulate_error) {}

  std::exception_ptr checkForNCCLErrors(
      const std::vector<std::shared_ptr<c10d::NCCLComm>>& ncclComms)
      const override {
    if (simulateError_) {
      return std::make_exception_ptr(std::runtime_error("Error"));
    }
    return c10d::ProcessGroupNCCL::WorkNCCL::checkForNCCLErrors(ncclComms);
  }

 private:
  bool simulateError_;
};

class ProcessGroupNCCLSimulateErrors : public c10d::ProcessGroupNCCL {
 public:
  ProcessGroupNCCLSimulateErrors(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> opts)
      : ProcessGroupNCCL(store, rank, size, opts), simulateError_(false) {}

  std::exception_ptr checkForNCCLErrors(
      const std::vector<std::shared_ptr<c10d::NCCLComm>>& ncclComms) override {
    if (simulateError_) {
      return std::make_exception_ptr(std::runtime_error("Error"));
    }
    return c10d::ProcessGroupNCCL::checkForNCCLErrors(ncclComms);
  }

  std::chrono::duration<int64_t, std::milli> getWatchdogSleepInterval() {
    return std::chrono::milliseconds(
        ProcessGroupNCCLSimulateErrors::kWatchdogThreadSleepMillis);
  }

  c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL> initWork(
      std::vector<at::Device> devices,
      int rank,
      c10d::OpType opType,
      const char* profilingTitle,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {}) override {
    return c10::make_intrusive<WorkNCCLSimulateErrors>(
        devices, simulateError_, rank, opType, seq_);
  }

  size_t getNCCLCommCacheSize() {
    return devNCCLCommMap_.size();
  }

  void simulateError() {
    simulateError_ = true;
  }

  void resetError() {
    simulateError_ = false;
  }

 private:
  bool simulateError_;
};

class WorkNCCLTimedoutErrors : public c10d::ProcessGroupNCCL::WorkNCCL {
 public:
  WorkNCCLTimedoutErrors(
      const std::vector<at::Device>& devices,
      bool set_timedout_error,
      int rank,
      c10d::OpType opType,
      uint64_t seq)
      : WorkNCCL(devices, rank, opType, seq),
        setTimedoutError_(set_timedout_error) {}

 private:
  bool isCompleted() override {
    if (setTimedoutError_) {
      return false;
    }
    return c10d::ProcessGroupNCCL::WorkNCCL::isCompleted();
  }

 private:
  bool setTimedoutError_;
};

class ProcessGroupNCCLTimedOutErrors : public ProcessGroupNCCLSimulateErrors {
 public:
  ProcessGroupNCCLTimedOutErrors(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> opts)
      : ProcessGroupNCCLSimulateErrors(store, rank, size, opts),
        watchDogDebugInfoFinished_(false),
        setTimedoutError_(false) {}

  c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL> initWork(
      std::vector<at::Device> devices,
      int rank,
      c10d::OpType opType,
      const char* profilingTitle,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {}) override {
    return c10::make_intrusive<WorkNCCLTimedoutErrors>(
        devices, setTimedoutError_, rank, opType, seq_);
  }

  void setTimedoutError() {
    setTimedoutError_ = true;
  }

  void resetTimedoutError() {
    setTimedoutError_ = false;
  }

  bool getWatchDogDebugInfoFinishedFlag() {
    return watchDogDebugInfoFinished_;
  }

  // In the constructor of ProcessGroupNCCL. We don't allow the watchdog thread
  // to run any handling or desync report when the main thread is block wait.
  // Even if users set handling and turn on desyncDebug flag, they will get
  // reset. For the ease of unit test, we want the main thread to be block wait,
  // so we have this hack to manually set the desync debug flag after PG
  // creation.
  void forceSetDesyncDebugFlag() {
    desyncDebug_ = true;
  }

 protected:
  std::string getNCCLWatchdogDebugInfo() override {
    LOG(INFO) << "overridden getNCCLWatchdogDebugInfo called";
    watchDogDebugInfoFinished_ = true;
    return "";
  }
  bool watchDogDebugInfoFinished_;

 private:
  bool setTimedoutError_;
};

class ProcessGroupNCCLNoHeartbeatCaught
    : public ProcessGroupNCCLTimedOutErrors {
 public:
  ProcessGroupNCCLNoHeartbeatCaught(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> opts)
      : ProcessGroupNCCLTimedOutErrors(store, rank, size, opts),
        hasMonitorThreadCaughtError_(false) {}

  std::mutex& getWatchdogMutex() {
    return workMetaListMutex_;
  }

  bool getErrorCaughtFlag() {
    return hasMonitorThreadCaughtError_;
  }

  void forceTryWriteDebugInfo() {
    std::future<bool> asyncDebugDump = launchAsyncDebugDump();
    asyncDebugDump.wait();
  }

 protected:
  // Override the heartbeat monitor function to make sure that we capture
  // the exception in the monitor thread because we cannot try-catch it in
  // the main thread and we set a flag for the main thread to check.
  void heartbeatMonitor() override {
    try {
      c10d::ProcessGroupNCCL::heartbeatMonitor();
    } catch (std::runtime_error& e) {
      hasMonitorThreadCaughtError_ = true;
    }
  }

  // It's really hard to unit test std::abort. So we override it instead.
  // Commented this override, we do see process aborted with core dump without
  // this override.
  void terminateProcess(std::string errMsg) override {
    throw std::runtime_error(errMsg);
  }

  bool hasMonitorThreadCaughtError_;
};

class ProcessGroupNCCLDebugInfoStuck
    : public ProcessGroupNCCLNoHeartbeatCaught {
 public:
  ProcessGroupNCCLDebugInfoStuck(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> opts)
      : ProcessGroupNCCLNoHeartbeatCaught(store, rank, size, opts) {}

 protected:
  // Override the heartbeat monitor function to set a long timeout to mimic the
  // stuck in getting debug info.
  std::string getNCCLWatchdogDebugInfo() override {
    std::this_thread::sleep_for(
        std::chrono::seconds(heartbeatTimeoutInSec_ * 20));
    watchDogDebugInfoFinished_ = true;
    return "";
  }
};

class ProcessGroupNCCLErrorsTest : public ::testing::Test {
 protected:
  bool skipTest() {
    if (cudaNumDevices() == 0) {
      LOG(INFO) << "Skipping test since CUDA is not available";
      return true;
    }
#ifdef USE_C10D_NCCL
    if (torch::cuda::nccl::version() < kNcclErrorHandlingVersion) {
      LOG(INFO) << "Skipping test since NCCL version is too old";
      return true;
    }
#endif
    return false;
  }

  void SetUp() override {
    // Enable LOG(INFO) messages.
    c10::initLogging();
    size_t numDevices = cudaNumDevices();
    TemporaryFile file;
    store_ = c10::make_intrusive<::c10d::FileStore>(file.path, 1);

    at::cuda::OptionalCUDAGuard deviceGuard;
    tensors_.resize(numDevices);
    for (const auto i : c10::irange(numDevices)) {
      deviceGuard.set_index(i);
      tensors_[i] = at::ones({3, 3}, at::kCUDA);
    }
  }

  void TearDown() override {
    ASSERT_TRUE(setenv(c10d::TORCH_NCCL_BLOCKING_WAIT[0].c_str(), "0", 1) == 0);
  }

  std::vector<at::Tensor> tensors_;
  c10::intrusive_ptr<::c10d::FileStore> store_;
};

TEST_F(ProcessGroupNCCLErrorsTest, testNCCLErrorsBlocking) {
  if (skipTest()) {
    return;
  }

  ASSERT_TRUE(setenv(c10d::TORCH_NCCL_BLOCKING_WAIT[0].c_str(), "1", 1) == 0);
  auto options = c10d::ProcessGroupNCCL::Options::create();
  options->timeout = std::chrono::milliseconds(1000);
  ProcessGroupNCCLSimulateErrors pg(store_, 0, 1, options);

  auto work = pg.allreduce(tensors_);
  work->wait();
  EXPECT_TRUE(work->isSuccess());
  EXPECT_EQ(1, pg.getNCCLCommCacheSize());

  // Now run all reduce with errors.
  pg.simulateError();
  work = pg.allreduce(tensors_);
  EXPECT_THROW(work->wait(), std::runtime_error);

  // Verify the work item failed.
  EXPECT_TRUE(work->isCompleted());
  EXPECT_FALSE(work->isSuccess());
  EXPECT_THROW(work->wait(), std::runtime_error);

  // Communicators might be aborted here, further operations would fail.
}

TEST_F(ProcessGroupNCCLErrorsTest, testNCCLTimedoutErrorsBlocking) {
  if (skipTest()) {
    return;
  }

  ASSERT_TRUE(setenv(c10d::TORCH_NCCL_BLOCKING_WAIT[0].c_str(), "1", 1) == 0);
  auto options = c10d::ProcessGroupNCCL::Options::create();
  options->timeout = std::chrono::milliseconds(3000);
  ProcessGroupNCCLTimedOutErrors pg(store_, 0, 1, options);

  auto work = pg.allreduce(tensors_);
  work->wait();
  EXPECT_TRUE(work->isSuccess());
  EXPECT_EQ(1, pg.getNCCLCommCacheSize());

  // Now run all reduce with errors.
  pg.setTimedoutError();
  work = pg.allreduce(tensors_);
  EXPECT_THROW(work->wait(), c10::DistBackendError);

  // Communicators might be aborted here, further operations would fail.
}

TEST_F(ProcessGroupNCCLErrorsTest, testNCCLErrorsNonBlocking) {
  if (skipTest()) {
    return;
  }

  auto options = c10d::ProcessGroupNCCL::Options::create();
  options->timeout = std::chrono::milliseconds(3000);
  ProcessGroupNCCLSimulateErrors pg(store_, 0, 1, options);

  auto work = pg.allreduce(tensors_);
  pg.barrier()->wait();
  EXPECT_TRUE(work->isSuccess());
  EXPECT_EQ(1, pg.getNCCLCommCacheSize());

  // Now run all reduce with errors.
  pg.simulateError();
  work = pg.allreduce(tensors_);

  // Should not throw exceptions.
  work->wait();
  pg.barrier()->wait();

  // Verify the work item failed.
  EXPECT_TRUE(work->isCompleted());
  EXPECT_FALSE(work->isSuccess());

  // Communicators might be aborted here, further operations would fail.
}

// Function to read what we wrote to the local disk for validation.
std::string readTraceFromFile(const std::string& filename, size_t size) {
  std::ifstream file(filename, std::ios::binary);
  // Read the strings from the file
  if (file) { // While the file stream is in good state
    std::string str(size, '\0');
    file.read(&str[0], size);
    if (file) {
      return str;
    }
  }
  return "";
}

// Extend the nested class outside the parent class
class TestDebugInfoWriter : public c10d::DebugInfoWriter {
 public:
  TestDebugInfoWriter() : DebugInfoWriter(0, 0) {}

  void write(const std::string& ncclTrace) override {
    traces_.assign(ncclTrace.begin(), ncclTrace.end());
    c10d::DebugInfoWriter::write(ncclTrace);
  }

  std::vector<uint8_t>& getTraces() {
    return traces_;
  }

 private:
  std::vector<uint8_t> traces_;
};

TEST_F(ProcessGroupNCCLErrorsTest, testNCCLErrorsNoHeartbeat) {
  if (skipTest()) {
    return;
  }

  int heartBeatIntervalInSec = 2;
  std::string timeInterval = std::to_string(heartBeatIntervalInSec);
  ASSERT_TRUE(setenv(c10d::TORCH_NCCL_BLOCKING_WAIT[0].c_str(), "1", 1) == 0);
  ASSERT_TRUE(
      setenv(
          c10d::TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC[0].c_str(),
          timeInterval.c_str(),
          1) == 0);
  ASSERT_TRUE(
      setenv(c10d::TORCH_NCCL_ENABLE_MONITORING[0].c_str(), "1", 1) == 0);
  auto tempFilename = c10::str(
      std::filesystem::temp_directory_path().string(), "/nccl_trace_rank_");
  ASSERT_TRUE(
      setenv("TORCH_NCCL_DEBUG_INFO_TEMP_FILE", tempFilename.c_str(), 1) == 0);
  // Enable nccl flight recorder.
  ASSERT_TRUE(setenv("TORCH_NCCL_TRACE_BUFFER_SIZE", "10", 1) == 0);
  auto options = c10d::ProcessGroupNCCL::Options::create();
  // Set a long watchdog timeout, so that we have enough time to lock the
  // watchdog and let the heartbeat monitor thread to kick in.
  options->timeout = std::chrono::milliseconds(30000);
  ProcessGroupNCCLNoHeartbeatCaught pg(store_, 0, 1, options);
  // The storer here is very similar to the fallback storer.
  // The only difference is that we are storing traces also in memory for
  // validation.
  std::unique_ptr<TestDebugInfoWriter> wrterForTestPtr =
      std::make_unique<TestDebugInfoWriter>();
  std::vector<uint8_t>& traces = wrterForTestPtr->getTraces();
  pg.registerDebugInfoWriter(std::move(wrterForTestPtr));

  // Normal collective case.
  auto work = pg.allreduce(tensors_);
  work->wait();
  EXPECT_TRUE(work->isSuccess());

  work = pg.allreduce(tensors_);
  {
    // Now run all reduce with errors.
    std::lock_guard<std::mutex> lock(pg.getWatchdogMutex());
    LOG(INFO) << "Lock watchdog thread.";
    // Wait long enough before monitor thread throws exceptions.
    std::this_thread::sleep_for(
        std::chrono::seconds(heartBeatIntervalInSec * 3));
    // Check the monitoring thread launched and exception thrown.
    EXPECT_TRUE(pg.getErrorCaughtFlag());
  }
  work->wait();
  EXPECT_TRUE(work->isSuccess());
  EXPECT_TRUE(traces.size() > 0);
  auto filename = c10::str(tempFilename, 0, "_", 0);
  auto traceFromStorage = readTraceFromFile(filename, traces.size());
  // Check the traces read from storage match with the original nccl trace.
  EXPECT_TRUE(traceFromStorage == std::string(traces.begin(), traces.end()));
  std::filesystem::remove(filename);
}

class ProcessGroupNCCLWatchdogTimeoutTest : public ProcessGroupNCCLErrorsTest {
 protected:
  void SetUp() override {
    ProcessGroupNCCLErrorsTest::SetUp();
    std::string timeInterval = std::to_string(heartBeatIntervalInSec);
    ASSERT_TRUE(setenv(c10d::TORCH_NCCL_BLOCKING_WAIT[0].c_str(), "1", 1) == 0);
    ASSERT_TRUE(
        setenv(
            c10d::TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC[0].c_str(),
            timeInterval.c_str(),
            1) == 0);
    ASSERT_TRUE(
        setenv(c10d::TORCH_NCCL_ENABLE_MONITORING[0].c_str(), "1", 1) == 0);
    ASSERT_TRUE(setenv(c10d::TORCH_NCCL_DESYNC_DEBUG[0].c_str(), "1", 1) == 0);
    // We cannot capture the exception thrown in watchdog thread without making
    // lots of changes to the code. So we don't let the watchdog throw
    // exception.
    ASSERT_TRUE(
        setenv(c10d::TORCH_NCCL_ASYNC_ERROR_HANDLING[0].c_str(), "0", 1) == 0);
    options_ = c10d::ProcessGroupNCCL::Options::create();
    // Set a super short watchdog timeout.
    options_->timeout = std::chrono::milliseconds(100);
  }

  void watchdogTimeoutTestCommon(
      ProcessGroupNCCLNoHeartbeatCaught& pg,
      int multiplier) {
    pg.forceSetDesyncDebugFlag();
    pg.setTimedoutError();
    auto work = pg.allreduce(tensors_);
    std::this_thread::sleep_for(
        std::chrono::seconds(heartBeatIntervalInSec * multiplier));
    EXPECT_THROW(work->wait(), c10::DistBackendError);
  }

  const int heartBeatIntervalInSec = 2;
  c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> options_;
};

TEST_F(ProcessGroupNCCLWatchdogTimeoutTest, testNCCLTimedoutDebugInfoFinished) {
  // Need to have this check for every test to make sure we only run the test
  // when there are GPUs available.
  if (skipTest()) {
    return;
  }

  ProcessGroupNCCLNoHeartbeatCaught pg(store_, 0, 1, options_);
  // Write debug info will lead to watchdog thread to wait for 30 seconds.
  // And this is hard to override, so we just call it before hand. Otherwise,
  // we need to set a long heartbeat timeout which will make the test way
  // slower.
  pg.forceTryWriteDebugInfo();
  watchdogTimeoutTestCommon(pg, 2);

  // The flag is true shows that the heartbeat monitor thread does not kill
  // the watchdog thread when it is getting debug info such as desync debug
  // info.
  EXPECT_TRUE(pg.getWatchDogDebugInfoFinishedFlag());
  // The flag is false shows that the heartbeat monitor thread does not
  // trigger process abort if getting debug info and destroy PG is fast.
  EXPECT_FALSE(pg.getErrorCaughtFlag());

  // Communicators might be aborted here, further operations would fail.
}

TEST_F(ProcessGroupNCCLWatchdogTimeoutTest, testNCCLTimedoutDebugInfoStuck) {
  // Need to have this check for every test to make sure we only run the test
  // when there are GPUs available.
  if (skipTest()) {
    return;
  }
  ProcessGroupNCCLDebugInfoStuck pg(store_, 0, 1, options_);
  // Need to keep main thread sleep longer so that we can let heartbeat monitor
  // thread to finish the extra wait and flip the flag.
  watchdogTimeoutTestCommon(pg, 4);
  // The flag is false shows that we get stuck in getting debug info such as
  // desync debug info in the watchdog thread.
  EXPECT_FALSE(pg.getWatchDogDebugInfoFinishedFlag());
  // The flag is true shows that the heartbeat monitor thread does trigger
  // process abort if getting debug info gets stuck.
  EXPECT_TRUE(pg.getErrorCaughtFlag());

  // Communicators might be aborted here, further operations would fail.
}
