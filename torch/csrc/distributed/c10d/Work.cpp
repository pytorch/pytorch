#include <ATen/ThreadLocalState.h>
#include <distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/RankLocal.hpp>
#include <torch/csrc/distributed/c10d/cuda/StreamBlock.hpp>

#include <torch/csrc/distributed/c10d/Work.hpp>
#include <utility>

namespace {

class WorkRegistry {
 public:
  void register_work(
      const at::Tensor& tensor,
      const c10::intrusive_ptr<c10d::Work>& work) {
    if (!tensor.has_storage()) {
      TORCH_WARN_ONCE(
          "Registering collective work for tensor without storage is not supported. "
          "Calling c10d_functional.wait_tensor() on this tensor will not wait for the collective to complete. "
          "Unsupported tensor type: " +
          tensor.toString());
      return;
    }
    auto storage = tensor.storage().getWeakStorageImpl();
    std::unique_lock lock(lock_);

    auto it = registry_.find(storage);
    if (it == registry_.end()) {
      registry_.emplace(
          std::move(storage),
          std::vector<c10::intrusive_ptr<c10d::Work>>{work});
    } else {
      // There is no guarantee that the previous work object for this
      // tensor storage is completed before the new work object is registered.
      // Therefore we need to maintain a list of work objects for each tensor
      // storage.

      // Check if work is already in the list
      bool work_exists = false;
      for (const auto& existing_work : it->second) {
        if (existing_work == work) {
          work_exists = true;
          break;
        }
      }

      // Only append if work is not already in the list
      if (!work_exists) {
        it->second.push_back(work);
      }
    }
  }

  std::vector<c10::intrusive_ptr<c10d::Work>> pop_works(
      const at::Tensor& tensor) {
    const auto storage = tensor.storage().getWeakStorageImpl();
    std::unique_lock lock(lock_);
    auto it = registry_.find(storage);
    if (it == registry_.end()) {
      return {};
    }
    auto works = it->second;
    registry_.erase(it);
    return works;
  }

  void unregister_work(const c10::intrusive_ptr<c10d::Work>& work) {
    std::unique_lock lock(lock_);
    for (auto it = registry_.begin(); it != registry_.end();) {
      std::vector<c10::intrusive_ptr<c10d::Work>> nonmatching_works;
      for (const auto& _work : it->second) {
        if (_work != work) {
          nonmatching_works.push_back(_work);
        }
      }
      if (nonmatching_works.empty()) {
        it = registry_.erase(it);
      } else {
        it->second = std::move(nonmatching_works);
        ++it;
      }
    }
  }

  size_t get_work_registry_size() {
    std::unique_lock lock(lock_);
    size_t total_size = 0;
    for (const auto& [storage, works] : registry_) {
      total_size += works.size();
    }
    return total_size;
  }

  void set_allow_inflight_collective_as_graph_input(bool value) {
    std::unique_lock lock(lock_);
    allow_inflight_collective_as_graph_input_ = value;
  }

  bool allow_inflight_collective_as_graph_input() {
    std::unique_lock lock(lock_);
    return allow_inflight_collective_as_graph_input_;
  }

  ~WorkRegistry() {
    // If there are still unwaited work objects, their corresponding process
    // groups should have already been destroyed at this stage. Any attempts to
    // wait for these work objects or to destroy them will only result in
    // confusing errors. Therefore, we simply issue a warning and intentionally
    // allow the unwaited work objects to leak.
    size_t registry_size = get_work_registry_size();
    if (registry_size > 0) {
      TORCH_WARN(
          "At the time of process termination, there are still ",
          registry_size,
          " unwaited collective calls. "
          "Please review your program to ensure that:\n"
          "1. c10d_functional.wait_tensor() is invoked on all tensors returned from c10d_functional collective,\n"
          "2. c10d_functional.wait_tensor() is invoked on all output tensors of async_op=True torch.distributed collective "
          "called under `with allow_inflight_collective_as_graph_input_ctx():`,\n"
          "before the output tensors of the collective are used.");
    }
    for (auto& it : registry_) {
      for (auto& work : it.second) {
        work.release();
      }
    }
  }

 private:
  std::unordered_map<
      c10::weak_intrusive_ptr<c10::StorageImpl>,
      std::vector<c10::intrusive_ptr<c10d::Work>>>
      registry_;
  bool allow_inflight_collective_as_graph_input_ = false;
  std::mutex lock_;
};

static WorkRegistry process_registry;

} // namespace

namespace c10d {

Work::Work(
    int rank,
    OpType opType,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors)
    : rank_(rank), opType_(opType) {
  if (profilingTitle != nullptr) {
    auto recordingFunction =
        std::make_shared<at::RecordFunction>(at::RecordScope::USER_SCOPE);
    if (recordingFunction->isActive()) {
      // Work events follow a future like pattern and can potentially be marked
      // as complete by different threads, so explicitly set as async event.
      recordingFunction->_setAsync();
      // Passing input tensor to recordFunction allows for shape information in
      // profiling output.
      std::vector<c10::IValue> inputs;
      if (inputTensors) {
        inputs.reserve(inputTensors->size());
        for (const auto& tensor : *inputTensors) {
          inputs.emplace_back(tensor);
        }
      }
      recordingFunction->before(
          profilingTitle,
          c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()));
      std::function<void()> end_handler = [recordingFunction]() {
        recordingFunction->end();
      };
      recordFunctionEndCallback_ = at::wrapPropagateTLSState(end_handler);
    }
  }
}

OpType Work::retrieveOpType() const {
  return opType_;
}

Work::~Work() = default;

bool Work::isCompleted() {
  std::lock_guard<std::mutex> lock(mutex_);
  return completed_;
}

bool Work::isSuccess() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !exception_;
}

std::exception_ptr Work::exception() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return exception_;
}

int Work::sourceRank() const {
  TORCH_CHECK(
      false,
      "sourceRank() may only be called on work objects "
      "that correspond to a recv or recv-from-any call.");
}

std::vector<at::Tensor> Work::result() {
  TORCH_CHECK(false, "result() not implemented.");
}

void Work::synchronize() {
  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(
        c10::intrusive_ptr<Work>::unsafe_reclaim_from_nonowning(this));
  }
}

bool Work::wait(std::chrono::milliseconds timeout) {
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
      TORCH_CHECK(false, "Operation timed out!");
    }
  }
  if (exception_) {
    std::rethrow_exception(exception_);
  }
  synchronize();
  // Always return true, because abort API is not implemented.
  return true;
}

void Work::blockCurrentStream() {
  // block cuda stream indefinitely until work is completed.
  std::shared_ptr<c10d::cuda::StreamBlock> handle =
      c10d::cuda::block_stream(std::chrono::milliseconds(0));

  getFuture()->addCallback(
      [handle](c10::ivalue::Future& future) { handle->abort(); });
}

void Work::abort() {
  TORCH_CHECK(false, "Work::abort not implemented.");
}

c10::intrusive_ptr<c10::ivalue::Future> Work::getFuture(){
    TORCH_CHECK(false, "Work::getFuture not implemented.")}

c10::intrusive_ptr<c10::ivalue::Future> Work::getFutureResult() {
  TORCH_CHECK(false, "Work::getFutureResult not implemented.")
}

void Work::finish(std::exception_ptr exception) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_ = true;
  exception_ = std::move(exception);
  if (recordFunctionEndCallback_) {
    recordFunctionEndCallback_();
    recordFunctionEndCallback_ = nullptr;
  }
  lock.unlock();
  cv_.notify_all();
}

void Work::finishAndThrow(std::exception_ptr exception) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_ = true;
  exception_ = std::move(exception);
  if (recordFunctionEndCallback_) {
    recordFunctionEndCallback_();
    recordFunctionEndCallback_ = nullptr;
  }
  if (exception_) {
    std::rethrow_exception(exception_);
  }
}

float Work::getDuration() const {
  TORCH_CHECK(false, "This Backend doesn't support getDuration.");
}

uint64_t Work::getSequencenumber() const {
  TORCH_CHECK(false, "This Backend doesn't support getSequencenumber.");
}

class FutureWrappingWork : public Work {
 public:
  FutureWrappingWork(c10::intrusive_ptr<c10::ivalue::Future> fut)
      : _fut(std::move(fut)) {}

  ~FutureWrappingWork() override = default;

  bool isCompleted() override {
    return _fut->completed();
  }

  bool isSuccess() const override {
    return _fut->hasValue();
  }

  std::exception_ptr exception() const override {
    return _fut->exception_ptr();
  }

  int sourceRank() const override {
    TORCH_CHECK(false, "FutureWrappingWork::sourceRank() not implemented");
  }

  std::vector<at::Tensor> result() override {
    return _fut->value().toPyObjectHolder()->extractTensors();
  }

  bool wait(std::chrono::milliseconds timeout) override {
    // FIXME
    TORCH_CHECK(
        timeout == kNoTimeout,
        "FutureWrappingWork::wait() with finite timeout not implemented");
    _fut->wait();
    return true;
  }

  void abort() override {
    TORCH_CHECK(false, "FutureWrappingWork::abort() not implemented");
  }

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    return _fut;
  }

 private:
  c10::intrusive_ptr<c10::ivalue::Future> _fut;
};

c10::intrusive_ptr<Work> Work::create_from_future(
    const c10::intrusive_ptr<c10::ivalue::Future>& future) {
  return c10::make_intrusive<FutureWrappingWork>(future);
}

void register_work(
    const at::Tensor& tensor,
    const c10::intrusive_ptr<c10d::Work>& work) {
  RankLocal<WorkRegistry>::get().register_work(tensor, work);
}

at::Tensor wait_tensor(const at::Tensor& tensor) {
  auto works = RankLocal<WorkRegistry>::get().pop_works(tensor);
  for (const auto& work : works) {
    work->wait();
  }
  return tensor;
}

void unregister_work(const c10::intrusive_ptr<c10d::Work>& work) {
  RankLocal<WorkRegistry>::get().unregister_work(work);
}

size_t get_work_registry_size() {
  return RankLocal<WorkRegistry>::get().get_work_registry_size();
}

void set_allow_inflight_collective_as_graph_input(bool value) {
  return RankLocal<WorkRegistry>::get()
      .set_allow_inflight_collective_as_graph_input(value);
}

bool allow_inflight_collective_as_graph_input() {
  return RankLocal<WorkRegistry>::get()
      .allow_inflight_collective_as_graph_input();
}

} // namespace c10d
