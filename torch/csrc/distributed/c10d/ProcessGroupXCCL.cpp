#include <torch/csrc/distributed/c10d/ProcessGroupXCCL.hpp>
#include <fstream>
#include <mutex>
#include <sstream>

#ifdef USE_C10D_XCCL
#include <exception>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/torch.h>

namespace c10d {

namespace {
std::map<c10d::ReduceOp, ccl::reduction> xcclOps =
  {
    {ReduceOp::MIN, ccl::reduction::min},
    {ReduceOp::MAX, ccl::reduction::max},
    {ReduceOp::SUM, ccl::reduction::sum},
    {ReduceOp::PRODUCT, ccl::reduction::prod},
  };

std::map<at::ScalarType, ccl::datatype> xcclDatatypes =
  {
    {at::kByte, ccl::datatype::uint8},
    {at::kChar, ccl::datatype::int8},
    {at::kShort, ccl::datatype::int16},
    {at::kInt, ccl::datatype::int32},
    {at::kLong, ccl::datatype::int64},
    {at::kHalf, ccl::datatype::float16},
    {at::kFloat, ccl::datatype::float32},
    {at::kDouble, ccl::datatype::float64},
    {at::kBFloat16, ccl::datatype::bfloat16},
    {at::kBool, ccl::datatype::uint8},
  };

void check_gpu_single_tensor(
    const at::Tensor& tensor
) {
  if (!tensor.is_xpu() || tensor.is_sparse()) {
    C10_THROW_ERROR(ValueError, "Tensors must be XPU and dense");
  }
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
      C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
    }
  }
}

} // namespace

namespace {

ProcessGroupXCCL::WorkXCCL::WorkXCCL(std::vector<std::vector<at::Tensor>> outputTensors,
                                            int rank,
                                            c10d::OpType opType,
                                            const c10::optional<std::vector<at::Tensor>>& inputTensors)
        : Work(rank, opType, nullptr, inputTensors),
          outputTensors_(std::move(outputTensors)),
          future_(createFutureAsOutput(outputTensors)
          );

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupXCCL::WorkXCCL::getFuture() {
  return future_;
}

c10::intrusive_ptr<Backend> ProcessGroupXCCL::createProcessGroupXCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size)
{
  return c10::make_intrusive<ProcessGroupXCCL>(store, rank, size);
}

c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL> ProcessGroupNCCL::initWork(
    at::Device& device,
    int rank,
    OpType opType,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs,
    bool record) {
  auto r = c10::make_intrusive<ProcessGroupNCCL::WorkXCCL>(
      device,
      rank,
      opType,
      seqCollective_,
      profilingTitle,
      profilingTitle != nullptr ? std::optional<std::vector<at::Tensor>>(inputs)
                                : std::nullopt,
      desyncDebug_,
      enableTiming_.load(),
      dist_debug_level_);
  if (record) {
    bool isP2P = isP2POp(opType);
    r->trace_id_ = NCCLTraceBuffer::get()->record(
        local_id_,
        std::make_tuple(pg_uid_, pg_desc_),
        seqCollective_,
        seqP2P_,
        op_id_,
        profilingTitle ? profilingTitle : "",
        inputs,
        outputs,
        r->ncclStartEvent_.get(),
        r->ncclEndEvent_.get(),
        options_->timeout,
        pgStatus_,
        isP2P);
  }
  return r;
}

ProcessGroupXCCL::~ProcessGroupXCCL()
{
}

std::shared_ptr<XCCLComm> ProcessGroupXCCL::getXCCLComm(
    const std::string& deviceKey,
    at::Device& device) {

  if (deviceKey.empty()) {
    C10_THROW_ERROR(
        DistBackendError,
        "Not able to create/get the CCL Communicator since "
            "the devices are empty ");
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (devXCCLCommMap_.find(deviceKey) != devXCCLCommMap_.end()) {
      return devXCCLCommMap_[deviceKey];
    }
  }

  std::shared_ptr<XCCLComm> xcclComm;

  XCCL_KVS kvs = get_kvs(rank_, store_);

  int numRanks, rank;
  numRanks = getSize();
  rank = getRank();

  ccl::vector_class<ccl::pair_class<int, ccl::device>> devs_rank;
  c10::impl::VirtualGuardImpl impl(device.type());
  c10::Stream stream = impl.getStream(device);
  auto q = get_sycl_queue(stream);
  auto ctx = ccl::create_context(q.get_context());
  devs_rank.emplace_back(rank, ccl::create_device(q.get_device()));
  auto xcclComm = ccl::create_communicator(numRanks, devs_rank, ctx, kvs);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    inInitializationCommMap_.emplace(deviceKey, ncclComm);
  }

  auto it = inInitializationCommMap_.find(deviceKey);
  if (it != inInitializationCommMap_.end()) {
    devXCCLCommMap_.emplace(deviceKey, std::move(it->second));
    inInitializationCommMap_.erase(deviceKey);

    ncclCommDevIdxMapMutex.lock();
    ncclCommDevIdxMap.emplace(ncclComm, device.index());
    ncclCommDevIdxMapMutex.unlock();
  }

  it = devXCCLCommMap_.find(deviceKey);
  TORCH_INTERNAL_ASSERT(
      it != devXCCLCommMap_.end(), "Communicators not populated in cache!");

  return it->second;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupNCCL::collective(
    at::Tensor& input,
    at::Tensor& output,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType) {

  auto device = input.device();
  const auto key = std::to_string(device.index());
  auto ncclComm = getXCCLComm(key, device);

  std::vector<at::Tensor> inputs{input};
  std::vector<at::Tensor> outputs{output};

  auto work =
      initWork(device, rank_, opType, profilingTitle, inputs, outputs, enqueue);

  // Store references to outputs to be used by WorkNCCL::result and operator<<.
  work->outputs_ =
      std::make_shared<std::vector<at::Tensor>>(std::move(outputs));

  if (avoidRecordStreams) {
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>();
    work->stashed_for_allocator_safety_->push_back(input);
  }

  at::cuda::OptionalCUDAGuard gpuGuard;

  // Start event should only be recorded before the ncclGroupStart()
  if (work->timingEnabled_) {
    work->ncclStartEvent_->record(ncclStream);
  }

  pre(ncclStream, work);

  ncclComm_t comm = ncclComm->getNcclComm();

  // Both `inputs' and `outputs' are created on a worker stream and used in
  // different ncclStreams.  Hence, both must record the ncclStream to
  // prevent being freed before the collective finishes.
  //
  // We only record `inputs' here, and leave recording `outputs' to `fn' for
  // operations where `inputs' and `outputs' are not the same.
  //
  // See [Sync Streams].
  if (!avoidRecordStreams) {
    if (!input.is_sparse()) {
      c10::cuda::CUDACachingAllocator::recordStream(
          input.storage().data_ptr(), ncclStream);
    } else {
      // for sparse input case record streams on both index and value
      // tensors
      c10::cuda::CUDACachingAllocator::recordStream(
          input.values().storage().data_ptr(), ncclStream);
      c10::cuda::CUDACachingAllocator::recordStream(
          input.indices().storage().data_ptr(), ncclStream);
    }
  }
#ifndef NCCL_HAS_COMM_NONBLOCKING
  C10D_NCCL_CHECK(
      fn(input, output, comm, ncclStream),
      ncclComm->getNcclCommFailureReason());
#else
  C10D_NCCL_CHECK_TIMEOUT(
      fn(input, output, comm, ncclStream),
      comm,
      ncclComm->getNcclCommFailureReason());
#endif

  post(ncclStream, work);

  // End event should only be recorded after the ncclGroupEnd()
  if (!coalescing_state_) {
    work->ncclEndEvent_->record(ncclStream);
  }
  work->ncclComm_ = ncclComm;

  {
    c10::cuda::CUDAMultiStreamGuard streamGuard(ncclStream);
    std::vector<at::Device> devices{device};
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // Add a callback that runs profiling end callbacks. wrapCallback() in CUDA
    // future blocks the stream this callback runs on the corresponding
    // ncclEndEvents_ ensuring appropriate synchronization.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          // uses_future = false allows us to skip synchronization in
          // ivalue::Future, but is only valid as long as the lambda doesn't use
          // the "Future" argument.
          /*uses_future=*/false);
    }
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // Set appropriate work parameters.
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams;
  work->opTimeout_ = options_->timeout;
  work->store_ = store_;
  // Record size info for debug. We only record the size on the first device as
  // multi-device per process is deprecated
  work->numelIn_ = input.numel();
  work->numelOut_ = output.numel();

  // Notify graphs before we check the capture status preemptively
  at::cuda::CUDAGraph::inc_pending_event_queries();
  if (enqueue) {
    workEnqueue(work);
  } else {
    at::cuda::CUDAGraph::dec_pending_event_queries();
  }

  return work;
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce_impl(
    at::Tensor& tensor,
    const AllreduceOptions& opts) {
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp =
            getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
        return ncclAllReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            ncclDataType,
            ncclReduceOp,
            comm,
            stream.stream());
      },
      OpType::ALLREDUCE,
      "nccl:all_reduce");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce(
  std::vector<at::Tensor>& tensors,
  const AllreduceOptions& opts)
{
  TORCH_CHECK(tensors.size() == 1, "Expecting one tensor only but got multiple");
  auto tensor = tensors.back();
  check_gpu_single_tensor(tensor);
  if (opts.reduceOp == ReduceOp::SUM) {
    TORCH_CHECK(false, "Cannot use ReduceOp SUM with XPU")
  }
  return allreduce_impl(tensor, opts);
}


}

}