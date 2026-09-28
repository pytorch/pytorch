#include <gtest/gtest.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/HashStore.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace c10d::nccl2 {

class NCCL2ReconfigureContractTestAccess {
 public:
  static c10::intrusive_ptr<WorkNCCL> makeWork(
      ProcessGroupNCCL& pg,
      const at::Tensor& tensor,
      cudaStream_t stream) {
    return pg.createWork(stream, std::chrono::seconds(30), tensor);
  }

  static void record(WorkNCCL& work, std::string_view name = "test") {
    work.recordStart(name);
    work.recordEnd();
  }

  static void enqueue(
      ProcessGroupNCCL& pg,
      const c10::intrusive_ptr<WorkNCCL>& work,
      cudaStream_t stream) {
    pg.enqueueWork(work, stream);
  }

  static bool terminal(
      WorkNCCL& work,
      WorkNCCL::WorkStatus status,
      std::exception_ptr exception = nullptr) {
    return work.state_->setTerminalStatus(status, std::move(exception));
  }

  static void setGeneration(ProcessGroupNCCL& pg, int64_t uuid) {
    pg.reconfigure_uuid_ = uuid;
    pg.work_generation_state_ = std::make_shared<WorkGenerationState>();
  }

  static void cancelGeneration(ProcessGroupNCCL& pg, int64_t uuid) {
    pg.workq_.failPendingGeneration(uuid);
  }

  static void finalizeQueue(ProcessGroupNCCL& pg) {
    pg.workq_.finalize();
  }

  static size_t shelfUseCount(const WorkNCCL& work) {
    return work.inputTensors_.use_count();
  }

  static bool futureCompleted(const WorkNCCL& work) {
    return work.state_->futureWorkResult->completed();
  }

  static uint8_t futureResult(const WorkNCCL& work) {
    return work.state_->futureWorkResult->value().to<uint8_t>();
  }

  static void addFutureCallback(
      WorkNCCL& work,
      std::function<void(c10::ivalue::Future&)> callback) {
    work.state_->futureWorkResult->addCallback(
        std::move(callback),
        /*uses_future=*/false);
  }

  static void invalidateGeneration(ProcessGroupNCCL& pg) {
    auto generation = pg.work_generation_state_;
    std::lock_guard<std::mutex> lock(generation->mutex);
    generation->invalidated = true;
  }
};

} // namespace c10d::nccl2

namespace {

using c10d::nccl2::DefaultNcclApi;
using c10d::nccl2::NCCL2ReconfigureContractTestAccess;
using c10d::nccl2::ProcessGroupNCCL;
using c10d::nccl2::WorkNCCL;

struct RevokeQuery {
  ncclResult_t apiResult{ncclSuccess};
  ncclResult_t asyncResult{ncclSuccess};
};

struct RevokeScript {
  ncclResult_t revokeResult{ncclSuccess};
  std::vector<RevokeQuery> queries;
  bool repeatInProgress{false};
  bool blockRevoke{false};
};

class ScriptedNcclApi final : public DefaultNcclApi {
 public:
  void push(RevokeScript script) {
    std::lock_guard<std::mutex> lock(mutex_);
    scripts_.push_back(std::move(script));
  }

  bool waitUntilRevokeEntered(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [&] { return revoke_entered_; });
  }

  void releaseRevoke() {
    std::lock_guard<std::mutex> lock(mutex_);
    release_revoke_ = true;
    cv_.notify_all();
  }

  ncclResult_t commRevoke(ncclComm_t comm) override {
    std::unique_lock<std::mutex> lock(mutex_);
    if (scripts_.empty()) {
      lock.unlock();
      return DefaultNcclApi::commRevoke(comm);
    }
    active_script_ = std::move(scripts_.front());
    scripts_.erase(scripts_.begin());
    active_ = true;
    query_index_ = 0;
    revoke_entered_ = true;
    release_revoke_ = !active_script_.blockRevoke;
    cv_.notify_all();
    cv_.wait(lock, [&] { return release_revoke_; });
    return active_script_.revokeResult;
  }

  ncclResult_t commGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError)
      override {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!active_) {
      lock.unlock();
      return DefaultNcclApi::commGetAsyncError(comm, asyncError);
    }
    if (active_script_.repeatInProgress) {
      *asyncError = ncclInProgress;
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      return ncclSuccess;
    }
    if (query_index_ >= active_script_.queries.size()) {
      lock.unlock();
      return DefaultNcclApi::commGetAsyncError(comm, asyncError);
    }
    const auto query = active_script_.queries[query_index_++];
    *asyncError = query.asyncResult;
    return query.apiResult;
  }

  ncclResult_t commAbort(ncclComm_t comm) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      active_ = false;
      revoke_entered_ = false;
    }
    return DefaultNcclApi::commAbort(comm);
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<RevokeScript> scripts_;
  RevokeScript active_script_;
  size_t query_index_{0};
  bool active_{false};
  bool revoke_entered_{false};
  bool release_revoke_{false};
};

struct HostGate {
  std::mutex mutex;
  std::condition_variable cv;
  bool started{false};
  bool release{false};
};

void blockCudaStream(void* opaque) {
  auto& gate = *static_cast<HostGate*>(opaque);
  std::unique_lock<std::mutex> lock(gate.mutex);
  gate.started = true;
  gate.cv.notify_all();
  gate.cv.wait(lock, [&] { return gate.release; });
}

class NCCL2ReconfigureContractTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ::setenv("TORCH_NCCL_BLOCKING_WAIT", "1", 1);
    c10::cuda::CUDAGuard guard(at::Device(at::kCUDA, 0));
    options_ = ProcessGroupNCCL::Options::create();
    options_->enable_reconfigure = true;
    options_->timeout = std::chrono::seconds(10);
    store_ = c10::make_intrusive<::c10d::HashStore>();
    pg_ = c10::make_intrusive<ProcessGroupNCCL>(store_, 0, 1, options_);
    api_ = std::make_shared<ScriptedNcclApi>();
    pg_->setNcclApi(api_);
    reconfigure(1000, std::chrono::seconds(10));
  }

  void TearDown() override {
    if (pg_) {
      try {
        pg_->shutdown();
      } catch (...) {
      }
      pg_.reset();
    }
  }

  void reconfigure(int64_t uuid, std::chrono::milliseconds timeout) {
    ::c10d::ReconfigureOptions opts;
    opts.uuid = uuid;
    opts.handles =
        std::vector<::c10d::ReconfigureHandle>{pg_->get_reconfigure_handle()};
    opts.timeout = timeout;
    pg_->reconfigure(opts)->wait();
  }

  at::Tensor tensor() {
    return at::empty(
        {16}, at::TensorOptions().device(at::Device(at::kCUDA, 0)));
  }

  c10::intrusive_ptr<ProcessGroupNCCL::Options> options_;
  c10::intrusive_ptr<::c10d::Store> store_;
  c10::intrusive_ptr<ProcessGroupNCCL> pg_;
  std::shared_ptr<ScriptedNcclApi> api_;
};

TEST_F(
    NCCL2ReconfigureContractTest,
    QueueCancellationIsGenerationScopedAndPreservesTerminalWork) {
  c10::cuda::CUDAGuard guard(at::Device(at::kCUDA, 0));
  constexpr int64_t oldUuid = 2000;
  constexpr int64_t newUuid = 2001;
  NCCL2ReconfigureContractTestAccess::setGeneration(*pg_, oldUuid);
  auto stream = at::cuda::getDefaultCUDAStream(0).stream();

  auto pending =
      NCCL2ReconfigureContractTestAccess::makeWork(*pg_, tensor(), stream);
  auto completed =
      NCCL2ReconfigureContractTestAccess::makeWork(*pg_, tensor(), stream);
  NCCL2ReconfigureContractTestAccess::addFutureCallback(
      *completed, [raw = pg_.get()](c10::ivalue::Future&) {
        NCCL2ReconfigureContractTestAccess::invalidateGeneration(*raw);
      });
  ASSERT_TRUE(NCCL2ReconfigureContractTestAccess::terminal(
      *completed, WorkNCCL::WorkStatus::COMPLETED));
  NCCL2ReconfigureContractTestAccess::enqueue(*pg_, pending, stream);
  NCCL2ReconfigureContractTestAccess::enqueue(*pg_, completed, stream);

  NCCL2ReconfigureContractTestAccess::setGeneration(*pg_, newUuid);
  auto newGeneration =
      NCCL2ReconfigureContractTestAccess::makeWork(*pg_, tensor(), stream);
  NCCL2ReconfigureContractTestAccess::enqueue(*pg_, newGeneration, stream);

  std::atomic<int> callbackCount{0};
  NCCL2ReconfigureContractTestAccess::addFutureCallback(
      *pending,
      [&callbackCount, raw = pg_.get(), oldUuid](c10::ivalue::Future&) {
        ++callbackCount;
        NCCL2ReconfigureContractTestAccess::cancelGeneration(*raw, oldUuid);
      });

  NCCL2ReconfigureContractTestAccess::cancelGeneration(*pg_, oldUuid);
  NCCL2ReconfigureContractTestAccess::cancelGeneration(*pg_, oldUuid);

  EXPECT_EQ(pending->status(), WorkNCCL::WorkStatus::ERROR);
  EXPECT_NE(pending->exception(), nullptr);
  EXPECT_TRUE(NCCL2ReconfigureContractTestAccess::futureCompleted(*pending));
  EXPECT_EQ(
      NCCL2ReconfigureContractTestAccess::futureResult(*pending),
      static_cast<uint8_t>(::c10d::WorkResult::COMM_ERROR));
  EXPECT_EQ(completed->status(), WorkNCCL::WorkStatus::COMPLETED);
  EXPECT_EQ(newGeneration->status(), WorkNCCL::WorkStatus::NOT_STARTED);
  EXPECT_FALSE(
      NCCL2ReconfigureContractTestAccess::futureCompleted(*newGeneration));
  EXPECT_EQ(callbackCount.load(), 1);
  EXPECT_EQ(NCCL2ReconfigureContractTestAccess::shelfUseCount(*pending), 2);

  NCCL2ReconfigureContractTestAccess::finalizeQueue(*pg_);
  EXPECT_EQ(NCCL2ReconfigureContractTestAccess::shelfUseCount(*pending), 1);
}

TEST_F(NCCL2ReconfigureContractTest, CompletionAndInvalidationHaveOneWinner) {
  c10::cuda::CUDAGuard guard(at::Device(at::kCUDA, 0));
  auto stream = at::cuda::getDefaultCUDAStream(0).stream();
  for (int iteration = 0; iteration < 32; ++iteration) {
    const int64_t uuid = 3000 + iteration;
    NCCL2ReconfigureContractTestAccess::setGeneration(*pg_, uuid);
    auto work =
        NCCL2ReconfigureContractTestAccess::makeWork(*pg_, tensor(), stream);
    NCCL2ReconfigureContractTestAccess::enqueue(*pg_, work, stream);
    std::atomic<bool> start{false};
    bool completed = false;
    std::thread completion([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      completed = NCCL2ReconfigureContractTestAccess::terminal(
          *work, WorkNCCL::WorkStatus::COMPLETED);
    });
    std::thread transition([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      NCCL2ReconfigureContractTestAccess::invalidateGeneration(*pg_);
    });
    start.store(true, std::memory_order_release);
    completion.join();
    transition.join();

    NCCL2ReconfigureContractTestAccess::cancelGeneration(*pg_, uuid);
    const auto status = work->status();
    ASSERT_TRUE(
        status == WorkNCCL::WorkStatus::COMPLETED ||
        status == WorkNCCL::WorkStatus::ERROR);
    EXPECT_EQ(status == WorkNCCL::WorkStatus::COMPLETED, completed);
    EXPECT_TRUE(NCCL2ReconfigureContractTestAccess::futureCompleted(*work));
    EXPECT_EQ(
        NCCL2ReconfigureContractTestAccess::futureResult(*work),
        static_cast<uint8_t>(
            status == WorkNCCL::WorkStatus::COMPLETED
                ? ::c10d::WorkResult::SUCCESS
                : ::c10d::WorkResult::COMM_ERROR));
  }
  NCCL2ReconfigureContractTestAccess::finalizeQueue(*pg_);
}

TEST_F(
    NCCL2ReconfigureContractTest,
    PendingWorkFailsBeforeRevokeAndTensorShelfLivesUntilAbortSettles) {
  c10::cuda::CUDAGuard guard(at::Device(at::kCUDA, 0));
  HostGate gate;
  auto stream = at::cuda::getDefaultCUDAStream(0).stream();
  ASSERT_EQ(cudaLaunchHostFunc(stream, blockCudaStream, &gate), cudaSuccess);
  {
    std::unique_lock<std::mutex> lock(gate.mutex);
    ASSERT_TRUE(gate.cv.wait_for(
        lock, std::chrono::seconds(10), [&] { return gate.started; }));
  }

  auto work =
      NCCL2ReconfigureContractTestAccess::makeWork(*pg_, tensor(), stream);
  NCCL2ReconfigureContractTestAccess::record(*work);
  NCCL2ReconfigureContractTestAccess::enqueue(*pg_, work, stream);
  EXPECT_FALSE(work->isCompleted());

  RevokeScript script;
  script.revokeResult = ncclSuccess;
  script.blockRevoke = true;
  api_->push(std::move(script));
  ::c10d::ReconfigureOptions opts;
  opts.uuid = 4000;
  opts.handles =
      std::vector<::c10d::ReconfigureHandle>{pg_->get_reconfigure_handle()};
  opts.timeout = std::chrono::seconds(10);

  std::exception_ptr reconfigureError;
  std::thread transition([&] {
    c10::cuda::CUDAGuard threadGuard(at::Device(at::kCUDA, 0));
    try {
      pg_->reconfigure(opts)->wait();
    } catch (...) {
      reconfigureError = std::current_exception();
    }
  });
  const bool revokeEntered =
      api_->waitUntilRevokeEntered(std::chrono::seconds(10));
  const auto statusDuringRevoke = work->status();
  const bool futureDuringRevoke =
      NCCL2ReconfigureContractTestAccess::futureCompleted(*work);
  const auto resultDuringRevoke = futureDuringRevoke
      ? NCCL2ReconfigureContractTestAccess::futureResult(*work)
      : 0;
  const auto shelfRefsDuringRevoke =
      NCCL2ReconfigureContractTestAccess::shelfUseCount(*work);
  bool submissionRejected = false;
  try {
    std::vector<at::Tensor> tensors{tensor()};
    pg_->allreduce(tensors);
  } catch (const c10::Error& error) {
    submissionRejected = std::string(error.what()).find("being reconfigured") !=
        std::string::npos;
  }
  {
    std::lock_guard<std::mutex> lock(gate.mutex);
    gate.release = true;
    gate.cv.notify_all();
  }
  const auto streamStatus = cudaStreamSynchronize(stream);
  api_->releaseRevoke();
  transition.join();

  EXPECT_TRUE(revokeEntered);
  EXPECT_EQ(reconfigureError, nullptr);
  EXPECT_EQ(streamStatus, cudaSuccess);
  EXPECT_EQ(statusDuringRevoke, WorkNCCL::WorkStatus::ERROR);
  EXPECT_TRUE(futureDuringRevoke);
  EXPECT_EQ(
      resultDuringRevoke, static_cast<uint8_t>(::c10d::WorkResult::COMM_ERROR));
  EXPECT_EQ(shelfRefsDuringRevoke, 2);
  EXPECT_TRUE(submissionRejected);
  EXPECT_EQ(NCCL2ReconfigureContractTestAccess::shelfUseCount(*work), 1);
  EXPECT_TRUE(pg_->isInitialized());
}

TEST_F(
    NCCL2ReconfigureContractTest,
    RevokeAsyncOutcomesAreBoundedAndRecoverable) {
  c10::cuda::CUDAGuard guard(at::Device(at::kCUDA, 0));
  std::vector<RevokeScript> scripts;
  scripts.push_back({ncclSuccess, {}, false, false});
  scripts.push_back(
      {ncclInProgress, {{ncclSuccess, ncclSuccess}}, false, false});
  scripts.push_back(
      {ncclInProgress, {{ncclSuccess, ncclUnhandledCudaError}}, false, false});
  scripts.push_back({ncclSystemError, {}, false, false});
  scripts.push_back(
      {ncclInProgress, {{ncclSystemError, ncclSuccess}}, false, false});
  scripts.push_back({ncclInProgress, {}, true, false});

  int64_t uuid = 5000;
  for (auto& script : scripts) {
    HostGate gate;
    auto stream = at::cuda::getDefaultCUDAStream(0).stream();
    ASSERT_EQ(cudaLaunchHostFunc(stream, blockCudaStream, &gate), cudaSuccess);
    bool callbackStarted = false;
    {
      std::unique_lock<std::mutex> lock(gate.mutex);
      callbackStarted = gate.cv.wait_for(
          lock, std::chrono::seconds(10), [&] { return gate.started; });
    }
    if (!callbackStarted) {
      {
        std::lock_guard<std::mutex> lock(gate.mutex);
        gate.release = true;
        gate.cv.notify_all();
      }
      cudaStreamSynchronize(stream);
      ADD_FAILURE() << "CUDA stream gate did not start";
      return;
    }
    auto work =
        NCCL2ReconfigureContractTestAccess::makeWork(*pg_, tensor(), stream);
    NCCL2ReconfigureContractTestAccess::record(*work);
    NCCL2ReconfigureContractTestAccess::enqueue(*pg_, work, stream);
    const bool workWasPending = !work->isCompleted();
    if (!workWasPending) {
      {
        std::lock_guard<std::mutex> lock(gate.mutex);
        gate.release = true;
        gate.cv.notify_all();
      }
      cudaStreamSynchronize(stream);
      ADD_FAILURE() << "old Work was not pending before reconfigure";
      return;
    }

    script.blockRevoke = true;
    api_->push(std::move(script));
    const auto timeout = std::chrono::seconds(2);
    const auto currentUuid = uuid++;
    std::exception_ptr reconfigureError;
    std::thread transition([&] {
      c10::cuda::CUDAGuard threadGuard(at::Device(at::kCUDA, 0));
      try {
        reconfigure(currentUuid, timeout);
      } catch (...) {
        reconfigureError = std::current_exception();
      }
    });
    const bool revokeEntered =
        api_->waitUntilRevokeEntered(std::chrono::seconds(10));
    {
      std::lock_guard<std::mutex> lock(gate.mutex);
      gate.release = true;
      gate.cv.notify_all();
    }
    const auto streamStatus = cudaStreamSynchronize(stream);
    api_->releaseRevoke();
    transition.join();

    EXPECT_TRUE(revokeEntered);
    EXPECT_EQ(reconfigureError, nullptr);
    EXPECT_EQ(streamStatus, cudaSuccess);
    EXPECT_TRUE(pg_->isInitialized());
    EXPECT_EQ(work->status(), WorkNCCL::WorkStatus::ERROR);
    EXPECT_NE(work->exception(), nullptr);
    EXPECT_TRUE(NCCL2ReconfigureContractTestAccess::futureCompleted(*work));
    EXPECT_EQ(
        NCCL2ReconfigureContractTestAccess::futureResult(*work),
        static_cast<uint8_t>(::c10d::WorkResult::COMM_ERROR));
  }
}

} // namespace
