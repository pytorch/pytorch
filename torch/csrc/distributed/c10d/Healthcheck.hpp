#pragma once

#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include <c10/macros/Export.h>
#include <torch/custom_class.h>

namespace c10d {

class TORCH_API Healthcheck : public torch::CustomClassHolder {
 public:
  Healthcheck(
      c10::optional<int> exportOnError = c10::nullopt,
      std::chrono::milliseconds interval = std::chrono::seconds(10),
      std::chrono::milliseconds timeout = std::chrono::seconds(10));
  virtual ~Healthcheck() = default;

  virtual void shutdown();
  void wait();

  int getNumFailures() {
    return numFailures_;
  }

  // Calculate group rank and size information for the specific rank and world
  // size for the specific side. Side must be 0 or 1. Returns: (group,
  // groupRank, groupSize)
  static std::tuple<int, int, int> calculateGroupInfo(
      int side,
      int rank,
      int worldSize,
      int localWorldSize);

 protected:
  void waitFor(std::chrono::milliseconds duration);
  bool isShutdown();

 private:
  // Called to setup each side, this is run on the worker thread.
  virtual void setup(int side) = 0;

  // Called in an individual thread to run the healthcheck.
  virtual void runHealthcheck(int side) = 0;

  void runLoop();

 protected:
  const c10::optional<int> exitOnError_;
  const std::chrono::milliseconds interval_;
  const std::chrono::milliseconds timeout_;

 private:
  std::atomic<int> numFailures_{-1};
  std::future<void> worker_{};

  std::mutex shutdownM_;
  std::condition_variable shutdownCv_;
  bool shutdown_{false};
};

} // namespace c10d
