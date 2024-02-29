#pragma once
#include <c10/util/Exception.h>

#include <mutex>
#include <vector>

namespace torch {
namespace autograd {
namespace utils {

// Warning handler for multi-threaded contexts. Gather warnings from
// all threads into a single queue, then process together at the end
// in the main thread.
class DelayWarningHandler : public at::WarningHandler {
 public:
  ~DelayWarningHandler() override = default;
  void replay_warnings();

 private:
  void process(const c10::Warning& warning) override;

  std::vector<c10::Warning> warnings_;
  std::mutex mutex_;
};

} // namespace utils
} // namespace autograd
} // namespace torch
