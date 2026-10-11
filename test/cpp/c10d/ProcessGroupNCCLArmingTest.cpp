#include <chrono>
#include <cstdlib>
#include <future>

#include <gtest/gtest.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/distributed/c10d/HashStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

namespace {

using HeartbeatMonitor = c10d::ProcessGroupNCCL::HeartbeatMonitor;
using namespace std::chrono_literals;

// Model a monitor that is still executing but has left store polling, as the
// real monitor does during its post-dump wait. This tests the production
// arming method, not the path into that wait.
class NonPollingMonitor : public HeartbeatMonitor {
 public:
  explicit NonPollingMonitor(c10d::ProcessGroupNCCL* pg)
      : HeartbeatMonitor(pg),
        enteredFuture_(entered_.get_future()),
        release_(releasePromise_.get_future()) {}

  void waitUntilRunning() {
    enteredFuture_.wait();
  }

  void release() {
    releasePromise_.set_value();
  }

  void runLoop() override {
    entered_.set_value();
    release_.wait();
  }

 private:
  std::promise<void> entered_;
  std::future<void> enteredFuture_;
  std::promise<void> releasePromise_;
  std::future<void> release_;
};

class ExitedMonitor : public HeartbeatMonitor {
 public:
  explicit ExitedMonitor(c10d::ProcessGroupNCCL* pg)
      : HeartbeatMonitor(pg), exitedFuture_(exited_.get_future()) {}

  void waitUntilExited() {
    exitedFuture_.wait();
  }

  void runLoop() override {
    exited_.set_value_at_thread_exit();
  }

 private:
  std::promise<void> exited_;
  std::future<void> exitedFuture_;
};

TEST(ProcessGroupNCCLArmingTest, RequiresWorkerAcknowledgment) {
  if (!at::cuda::is_available()) {
    GTEST_SKIP() << "ProcessGroupNCCL construction requires a GPU";
  }

  // Do not start the PG's own watchdog/monitor; the test drives an independent
  // HeartbeatMonitor against the same default PG.
  ASSERT_EQ(setenv(c10d::TORCH_NCCL_BLOCKING_WAIT[0].c_str(), "1", 1), 0);
  ASSERT_EQ(setenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "480", 1), 0);
  ASSERT_EQ(setenv("TORCH_NCCL_DUMP_ON_TIMEOUT", "1", 1), 0);
  ASSERT_EQ(setenv("TORCH_NCCL_COORD_CHECK_MILSEC", "100", 1), 0);
  auto store = c10::make_intrusive<c10d::HashStore>();
  auto options = c10d::ProcessGroupNCCL::Options::create();
  auto pg = c10::make_intrusive<c10d::ProcessGroupNCCL>(store, 0, 1, options);
  // This test must construct the first ProcessGroupNCCL in its process:
  // the default-PG responder is selected by process-global UID 0. The
  // dedicated test binary is intentional; same-process gtest repeats are not.
  ASSERT_EQ(pg->getUid(), 0);

  NonPollingMonitor nonPolling(pg.get());
  nonPolling.start();
  nonPolling.waitUntilRunning();
  EXPECT_FALSE(nonPolling.monitorDumpSignalsDuringShutdown(20ms));
  nonPolling.release();
  nonPolling.join();

  ExitedMonitor exited(pg.get());
  exited.start();
  exited.waitUntilExited();
  EXPECT_FALSE(exited.monitorDumpSignalsDuringShutdown(20ms));
  exited.join();

  HeartbeatMonitor active(pg.get());
  active.start();
  EXPECT_TRUE(active.monitorDumpSignalsDuringShutdown(5s));
  active.stop();
  active.join();
  pg->shutdown();
}

} // namespace
