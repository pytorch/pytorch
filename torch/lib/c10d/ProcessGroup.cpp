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

bool ProcessGroup::Work::wait(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (timeout == kNoTimeout) {
    // This waits without a timeout.
    cv_.wait(lock, [&] { return completed_; });
  } else {
    // Waits for the user-provided timeout.
    cv_.wait_for(lock, timeout, [&] { return completed_; });
    if (!completed_) {
      // Throw exception if the wait operation timed out and the work was not
      // completed.
      throw std::runtime_error("Operation timed out!");
    }
  }
  if (exception_) {
    std::rethrow_exception(exception_);
  }
  synchronize();
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroup::Work::abort() {
  TORCH_CHECK(false, "ProcessGroup::Work::abort not implemented.");
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroup::Work::getFuture() {
  TORCH_CHECK(false, "ProcessGroup::Work::getFuture not implemented.")
}

void ProcessGroup::Work::finish(std::exception_ptr exception) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_ = true;
  exception_ = exception;
  lock.unlock();
  cv_.notify_all();
}

void ProcessGroup::Work::finishAndThrow(std::exception_ptr exception) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_ = true;
  exception_ = exception;
  if (exception_) {
    std::rethrow_exception(exception_);
  }
}

ProcessGroup::ProcessGroup(int rank, int size) : rank_(rank), size_(size) {
  C10_LOG_API_USAGE_ONCE("c10d.process_group");
}

ProcessGroup::~ProcessGroup() {}

// This is introduced so that implementors of ProcessGroup would not need to
// have this implmentation.
std::shared_ptr<ProcessGroup::Work> ProcessGroup::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* usused */,
    std::vector<at::Tensor>& /* usused */,
    const AllgatherOptions& /* usused */) {
  throw std::runtime_error(
      "no support for allgather_coalesced in this process group");
}

void ProcessGroup::checkSplitSizes(
    const std::vector<int64_t>& split_sizes,
    const at::Tensor& tensor,
    int group_size) {
  if (split_sizes.size() == 0) {
    TORCH_CHECK(
        tensor.size(0) % group_size == 0,
        "Tensor's dim 0 does not divide equally across group size");
  } else {
    TORCH_CHECK(
        split_sizes.size() == group_size,
        "Number of tensor splits not equal to group size");
    int sum = std::accumulate(split_sizes.begin(), split_sizes.end(), 0);
    TORCH_CHECK(
        sum == tensor.size(0), "Split sizes doesn't match total dim 0 size");
  }
}

int64_t ProcessGroup::computeLengthsAndOffsets(
    const std::vector<int64_t>& split_sizes,
    const at::Tensor& tensor,
    std::vector<int>* lengths,
    std::vector<int>* offsets) {
  int64_t group_size = lengths->size();
  bool equal_splits = false;
  int64_t dim0_size = tensor.size(0);
  int64_t row_size = (dim0_size ? tensor.numel() / dim0_size : 1);
  int64_t split_size = 0;
  int64_t offset = 0;

  if (split_sizes.size() == 0) {
    equal_splits = true;
    split_size = tensor.size(0) / group_size;
  }
  for (int i = 0; i < group_size; i++) {
    int64_t length = row_size * (equal_splits ? split_size : split_sizes[i]);
    TORCH_INTERNAL_ASSERT(
        length <= std::numeric_limits<int>::max() &&
            offset <= std::numeric_limits<int>::max(),
        "Length or offset larger than INT_MAX not supported");
    (*lengths)[i] = length;
    (*offsets)[i] = offset;
    offset += length;
  }
  return offset;
}

int64_t ProcessGroup::computeLengthsAndOffsets(
    const std::vector<at::Tensor>& tensors,
    std::vector<int>* lengths,
    std::vector<int>* offsets) {
  int64_t group_size = lengths->size();
  int64_t offset = 0;
  for (int i = 0; i < group_size; i++) {
    int64_t length = tensors[i].numel();
    TORCH_INTERNAL_ASSERT(
        length <= std::numeric_limits<int>::max() &&
            offset <= std::numeric_limits<int>::max(),
        "Length or offset larger than INT_MAX not supported");
    (*lengths)[i] = length;
    (*offsets)[i] = offset;
    offset += length;
  }
  return offset;
}

} // namespace c10d
