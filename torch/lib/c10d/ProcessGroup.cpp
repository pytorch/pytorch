#include <c10d/ProcessGroup.hpp>

#include <c10/util/Logging.h>

namespace c10d {

ProcessGroup::Work::~Work() {}

bool ProcessGroup::Work::isCompleted() {
  std::lock_guard<std::mutex> lock(mutex_);
  return completed_;
}

bool ProcessGroup::Work::isSuccess() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !exception_;
}

std::exception_ptr ProcessGroup::Work::exception() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return exception_;
}

int ProcessGroup::Work::sourceRank() const {
  throw std::runtime_error(
      "sourceRank() may only be called on work objects "
      "that correspond to a recv or recv-from-any call.");
}

std::vector<at::Tensor> ProcessGroup::Work::result() const {
  throw std::runtime_error("result() not implemented.");
}

void ProcessGroup::Work::synchronize() {}

bool ProcessGroup::Work::wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [&] { return completed_; });
  if (exception_) {
    std::rethrow_exception(exception_);
  }
  synchronize();
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroup::Work::abort() {
  TORCH_CHECK(false, "ProcessGroup::Work::abort not implemented.")
}

void ProcessGroup::Work::finish(std::exception_ptr exception) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_ = true;
  exception_ = exception;
  lock.unlock();
  cv_.notify_all();
}

ProcessGroup::ProcessGroup(int rank, int size) : rank_(rank), size_(size) {
  C10_LOG_API_USAGE_ONCE("c10d.process_group");
}

ProcessGroup::~ProcessGroup() {}

} // namespace c10d
