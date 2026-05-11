#if USE_DISTRIBUTED

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <c10/util/string_view.h>
#include <torch/headeronly/core/DeviceType.h>

#include "ProcessGroupOCCL.hpp"
namespace {

c10::intrusive_ptr<c10::ivalue::Future> createFuture(
    const std::vector<at::Tensor>& outputTensors) {
  std::vector<at::Device> devices;
  for (const auto& tensor : outputTensors) {
    if (!tensor.device().is_cpu()) {
      devices.push_back(tensor.device());
    }
  }
  return c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
}

#define CHECK_TENSOR(tensor)                                                    \
  do {                                                                          \
    if (tensor.defined()) {                                                     \
      TORCH_CHECK(                                                              \
          tensor.device().type() == c10::DeviceType::PrivateUse1,               \
          "ProcessGroupOCCL only supports OpenReg (PrivateUse1) tensors. Got ", \
          tensor.device().type(),                                               \
          " for argument '",                                                    \
          #tensor,                                                              \
          "' in ",                                                              \
          __func__,                                                             \
          ".");                                                                 \
    }                                                                           \
  } while (0)

#define CHECK_TENSOR_LIST(tensorList)                                             \
  do {                                                                            \
    for (const auto& tensor : tensorList) {                                       \
      if (tensor.defined()) {                                                     \
        TORCH_CHECK(                                                              \
            tensor.device().type() == c10::DeviceType::PrivateUse1,               \
            "ProcessGroupOCCL only supports OpenReg (PrivateUse1) tensors. Got ", \
            tensor.device().type(),                                               \
            " for argument '",                                                    \
            #tensorList,                                                          \
            "' in ",                                                              \
            __func__,                                                             \
            ".");                                                                 \
      }                                                                           \
    }                                                                             \
  } while (0)

#define CHECK_TENSOR_LIST_OF_LISTS(tensorListOfLists)                               \
  do {                                                                              \
    for (const auto& tensor_list : tensorListOfLists) {                             \
      for (const auto& tensor : tensor_list) {                                      \
        if (tensor.defined()) {                                                     \
          TORCH_CHECK(                                                              \
              tensor.device().type() == c10::DeviceType::PrivateUse1,               \
              "ProcessGroupOCCL only supports OpenReg (PrivateUse1) tensors. Got ", \
              tensor.device().type(),                                               \
              " for argument '",                                                    \
              #tensorListOfLists,                                                   \
              "' in ",                                                              \
              __func__,                                                             \
              ".");                                                                 \
        }                                                                           \
      }                                                                             \
    }                                                                               \
  } while (0)

} // namespace

namespace c10d {

// OpenRegWork ----------------------------------------------------------------

ProcessGroupOCCL::OpenRegWork::OpenRegWork(
    std::function<void()> fn,
    std::vector<at::Tensor> outputTensors,
    OpType opType,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors)
    : Work(-1, opType, profilingTitle, inputTensors),
      fn_(std::move(fn)),
      outputTensors_(std::move(outputTensors)),
      future_(createFuture(outputTensors_)) {}

void ProcessGroupOCCL::OpenRegWork::execute(
    const c10::intrusive_ptr<OpenRegWork>& work) {
  try {
    work->fn_();
  } catch (...) {
    work->finishWorkError(std::current_exception());
    return;
  }
  work->finishWork();
}

void ProcessGroupOCCL::OpenRegWork::finishWork() {
  future_->markCompleted(c10::IValue(outputTensors_));
  finish();
}

void ProcessGroupOCCL::OpenRegWork::finishWorkError(
    const std::exception_ptr& eptr) {
  future_->setError(eptr);
  finish(eptr);
}

std::vector<at::Tensor> ProcessGroupOCCL::OpenRegWork::result() {
  TORCH_CHECK(
      isCompleted(),
      "Work needs to be completed before calling result(). "
      "Should call wait() before result().");
  return outputTensors_;
}

c10::intrusive_ptr<c10::ivalue::Future>
ProcessGroupOCCL::OpenRegWork::getFuture() {
  return future_;
}

// Options ------------------------------------------------------------------

ProcessGroupOCCL::Options::Options(std::chrono::milliseconds timeout)
    : Backend::Options(OCCL_BACKEND_NAME, timeout) {}

// ProcessGroupOCCL ---------------------------------------------------------

ProcessGroupOCCL::ProcessGroupOCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(store),
      options_(std::move(options)) {
  threads_.resize(options_->threads);
  for (const auto i : c10::irange(threads_.size())) {
    threads_[i] = std::thread(&ProcessGroupOCCL::runLoop, this, i);
  }
}

ProcessGroupOCCL::~ProcessGroupOCCL() {
  {
    std::lock_guard<std::mutex> lock(workMutex_);
    stop_ = true;
  }
  workProduceCV_.notify_all();
  for (auto& thread : threads_) {
    thread.join();
  }
}

void ProcessGroupOCCL::runLoop(int /* workerIndex */) {
  std::unique_lock<std::mutex> lock(workMutex_);
  while (!stop_) {
    if (workQueue_.empty()) {
      workProduceCV_.wait(lock);
      continue;
    }
    auto work = std::move(workQueue_.front());
    workQueue_.pop_front();
    lock.unlock();
    OpenRegWork::execute(work);
    lock.lock();
  }
}

void ProcessGroupOCCL::enqueue(c10::intrusive_ptr<OpenRegWork> work) {
  {
    std::lock_guard<std::mutex> lock(workMutex_);
    workQueue_.push_back(std::move(work));
  }
  workProduceCV_.notify_one();
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& /* opts */) {
  CHECK_TENSOR_LIST(tensors);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, tensors, OpType::BROADCAST, "occl:broadcast", tensors);
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& /* opts */) {
  CHECK_TENSOR_LIST(tensors);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, tensors, OpType::ALLREDUCE, "occl:allreduce", tensors);
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::allreduce_sparse(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& /* opts */) {
  CHECK_TENSOR_LIST(tensors);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, tensors, OpType::ALLREDUCE, "occl:allreduce_sparse", tensors);
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& /* opts */) {
  CHECK_TENSOR_LIST(tensors);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, tensors, OpType::ALLREDUCE, "occl:allreduce_coalesced", tensors);
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& /* opts */) {
  CHECK_TENSOR_LIST(tensors);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, tensors, OpType::REDUCE, "occl:reduce", tensors);
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& /* opts */) {
  CHECK_TENSOR(outputTensor);
  CHECK_TENSOR(inputTensor);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, std::vector<at::Tensor>{outputTensor},
      OpType::REDUCE_SCATTER, "occl:_reduce_scatter_base",
      std::vector<at::Tensor>{inputTensor});
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& /* opts */) {
  CHECK_TENSOR(output_tensor);
  CHECK_TENSOR(input_tensor);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, std::vector<at::Tensor>{output_tensor},
      OpType::_ALLGATHER_BASE, "occl:_allgather_base",
      std::vector<at::Tensor>{input_tensor});
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& /* opts */) {
  CHECK_TENSOR_LIST_OF_LISTS(outputs);
  CHECK_TENSOR_LIST(inputs);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, inputs, OpType::ALLGATHER, "occl:allgather", inputs);
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& output_lists,
    std::vector<at::Tensor>& input_list,
    const AllgatherOptions& /* opts */) {
  CHECK_TENSOR_LIST_OF_LISTS(output_lists);
  CHECK_TENSOR_LIST(input_list);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, input_list, OpType::ALLGATHER, "occl:allgather_coalesced",
      input_list);
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& /* opts */) {
  CHECK_TENSOR_LIST(outputs);
  CHECK_TENSOR_LIST(inputs);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, outputs, OpType::ALLGATHER, "occl:allgather_into_tensor_coalesced",
      inputs);
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const GatherOptions& /* opts */) {
  CHECK_TENSOR_LIST_OF_LISTS(outputs);
  CHECK_TENSOR_LIST(inputs);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, inputs, OpType::GATHER, "occl:gather", inputs);
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ScatterOptions& /* opts */) {
  CHECK_TENSOR_LIST(outputs);
  CHECK_TENSOR_LIST_OF_LISTS(inputs);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, outputs, OpType::SCATTER, "occl:scatter", outputs);
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::reduce_scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ReduceScatterOptions& /* opts */) {
  CHECK_TENSOR_LIST(outputs);
  CHECK_TENSOR_LIST_OF_LISTS(inputs);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, outputs, OpType::REDUCE_SCATTER, "occl:reduce_scatter", outputs);
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const ReduceScatterOptions& /* opts */) {
  CHECK_TENSOR_LIST(outputTensors);
  CHECK_TENSOR_LIST(inputTensors);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, outputTensors, OpType::REDUCE_SCATTER,
      "occl:reduce_scatter_tensor_coalesced", inputTensors);
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& /* outputCounts */,
    std::vector<int64_t>& /* inputCounts */,
    const AllToAllOptions& /* opts */) {
  CHECK_TENSOR(outputTensor);
  CHECK_TENSOR(inputTensor);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, std::vector<at::Tensor>{outputTensor},
      OpType::ALLTOALL_BASE, "occl:alltoall_base",
      std::vector<at::Tensor>{inputTensor});
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::send(
    std::vector<at::Tensor>& tensors,
    int /* dstRank */,
    int /* tag */) {
  CHECK_TENSOR_LIST(tensors);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, tensors, OpType::SEND, "occl:send", tensors);
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::recv(
    std::vector<at::Tensor>& tensors,
    int /* srcRank */,
    int /* tag */) {
  CHECK_TENSOR_LIST(tensors);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, tensors, OpType::RECV, "occl:recv", tensors);
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int /* tag */) {
  CHECK_TENSOR_LIST(tensors);
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, tensors, OpType::RECVANYSOURCE, "occl:recvAnysource", tensors);
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupOCCL::barrier(
    const BarrierOptions& /* opts */) {
  auto work = c10::make_intrusive<OpenRegWork>(
      []() {}, std::vector<at::Tensor>{}, OpType::BARRIER, "occl:barrier");
  enqueue(work);
  return work;
}

c10::intrusive_ptr<ProcessGroupOCCL> createProcessGroupOCCL(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout) {
  auto options = ProcessGroupOCCL::Options::create(
      std::chrono::milliseconds(
          static_cast<int64_t>(timeout.count() * 1000)));
  return c10::make_intrusive<ProcessGroupOCCL>(store, rank, size, options);
}

} // namespace c10d
#endif // USE_DISTRIBUTED
