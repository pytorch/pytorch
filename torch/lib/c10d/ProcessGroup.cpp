#include <c10d/ProcessGroup.hpp>

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

void ProcessGroup::Work::synchronize() {}

void ProcessGroup::Work::wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [&] { return completed_; });
  if (exception_) {
    std::rethrow_exception(exception_);
  }
  synchronize();
}

void ProcessGroup::Work::finish(std::exception_ptr exception) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_ = true;
  exception_ = exception;
  lock.unlock();
  cv_.notify_all();
}

ProcessGroup::ProcessGroup(int rank, int size) : rank_(rank), size_(size) {}

ProcessGroup::~ProcessGroup() {}

} // namespace c10d
