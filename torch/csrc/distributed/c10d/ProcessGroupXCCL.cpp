#ifdef USE_C10D_XCCL

#include <comm/XPUGuard.h>
#include <torch/csrc/distributed/c10d/ProcessGroupXCCL.hpp>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <c10/core/DeviceType.h>
#include <c10/util/Optional.h>

namespace c10d {

namespace {
const std::map<c10d::ReduceOp, ccl::reduction> xcclOps = {
    {ReduceOp::MIN, ccl::reduction::min},
    {ReduceOp::MAX, ccl::reduction::max},
    {ReduceOp::SUM, ccl::reduction::sum},
    {ReduceOp::PRODUCT, ccl::reduction::prod},
};

const std::map<at::ScalarType, ccl::datatype> xcclDatatypes = {
    {at::kByte, ccl::datatype::uint8},
    {at::kChar, ccl::datatype::int8},
    {at::kInt, ccl::datatype::int32},
    {at::kLong, ccl::datatype::int64},
    {at::kHalf, ccl::datatype::float16},
    {at::kFloat, ccl::datatype::float32},
    {at::kDouble, ccl::datatype::float64},
    {at::kBFloat16, ccl::datatype::bfloat16},
    {at::kBool, ccl::datatype::uint8},
};

void checkXPUTensor(at::Tensor& tensor) {
  if (!tensor.is_xpu() || tensor.is_sparse() || tensor.is_complex()) {
    C10_THROW_ERROR(
        ValueError, "Tensors must be XPU and dense and non-complex");
    if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
      C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
    }
  }
}

ccl::datatype getXcclDataType(
    at::ScalarType type,
    bool is_reduction_op = false) {
  TORCH_CHECK(
      !isFloat8Type(type) && is_reduction_op,
      "Float8 dtypes are not currenlty supported for XCCL reductions");
  auto it = xcclDatatypes.find(type);
  TORCH_CHECK_WITH(
      TypeError,
      it != xcclDatatypes.end(),
      "Input tensor data type is not supported for XCCL process group: ",
      type);
  return it->second;
}

ccl::reduction getXcclReduceOp(const ReduceOp& reduceOp, at::Tensor& input) {
  try {
    if (input.scalar_type() == at::kBool && reduceOp == ReduceOp::SUM) {
      // Map sum to max for bool tensors to avoid overflow issues with sum.
      return ccl::reduction::max;
    }
    return xcclOps.at(reduceOp);
  } catch (const std::out_of_range&) {
    C10_THROW_ERROR(
        ValueError,
        "Cannot use ReduceOp." + reduceOpToString(reduceOp) + " with XCCL");
  }
}

void syncStream(
    at::Device& device,
    at::xpu::XPUEvent& xcclEvent,
    at::xpu::XPUStream& xcclStream) {
  xcclEvent.record(at::xpu::getCurrentXPUStream(device.index()));
  xcclEvent.block(xcclStream);
}
} // namespace

constexpr int64_t kSynchronizeBusyWaitMillis = 10;

ProcessGroupXCCL::WorkXCCL::WorkXCCL(
    at::Device& device,
    int rank,
    OpType opType,
    const std::optional<std::vector<at::Tensor>>& inputs)
    : Work(rank, opType, "profilingTitle", inputs),
      device_(device),
      workStartTime_(std::chrono::steady_clock::now()) {
  xcclEndEvent_ = std::make_shared<at::xpu::XPUEvent>();
}

ProcessGroupXCCL::WorkXCCL::WorkXCCL(const WorkXCCL& w)
    : Work(w.rank_, w.opType_),
      device_(w.device_),
      xcclEndEvent_(w.xcclEndEvent_),
      blockingWait_(w.blockingWait_),
      workStartTime_(w.workStartTime_) {}

ProcessGroupXCCL::WorkXCCL::~WorkXCCL() = default;

bool ProcessGroupXCCL::WorkXCCL::isCompleted() {
  if (xcclEndEvent_ && xcclEndEvent_->query()) {
    return true;
  }
  return false;
}

void ProcessGroupXCCL::WorkXCCL::synchronize() {
  synchronizeInternal(kNoTimeout);
}

void ProcessGroupXCCL::WorkXCCL::synchronizeInternal(
    std::chrono::milliseconds timeout) {
  auto currentStream = at::xpu::getCurrentXPUStream(device_.index());
  xcclEndEvent_->block(currentStream);
  if (blockingWait_) {
    while (!isCompleted()) {
      auto currentTimepoint = std::chrono::steady_clock::now();
      auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          currentTimepoint - workStartTime_);
      if (timeElapsed >= timeout) {
        std::string exceptionMsg = c10::str(
            "Work ran time out after ", timeElapsed.count(), " milliseconds.");
        TORCH_CHECK(false, exceptionMsg)
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
  }
}

bool ProcessGroupXCCL::WorkXCCL::wait(std::chrono::milliseconds timeout) {
  synchronizeInternal(timeout);
  return true;
}

ProcessGroupXCCL::ProcessGroupXCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size)
    : Backend(rank, size), store_(store) {
  blockingWait_ = getCvarBool(TORCH_XCCL_BLOCKING_WAIT, false);
  init();
}

ProcessGroupXCCL::~ProcessGroupXCCL() = default;

c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> ProcessGroupXCCL::initWork(
    at::Device& device,
    int rank,
    OpType opType,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs) {
  auto r = c10::make_intrusive<ProcessGroupXCCL::WorkXCCL>(
      device, rank, opType, std::optional<std::vector<at::Tensor>>(inputs));
  return r;
}

std::shared_ptr<xcclComm_t> ProcessGroupXCCL::getXCCLComm(
    const std::string& deviceKey,
    at::Device& device) {
  TORCH_CHECK_WITH(
      DistBackendError,
      !deviceKey.empty(),
      "Not able to create/get "
      "XCCL Communicator since the devices are empty ");
  {
    // todo: why do we need mutex here?
    std::lock_guard<std::mutex> lock(mutex_);
    if (devXCCLCommMap_.find(deviceKey) != devXCCLCommMap_.end()) {
      return devXCCLCommMap_[deviceKey];
    }
  }

  int numRanks, rank;
  numRanks = getSize();
  rank = getRank();

  c10::impl::VirtualGuardImpl impl(device.type());
  c10::Stream stream =
      impl.getStreamFromGlobalPool(device, /*isHighPriority=*/false);
  sycl::queue& q = c10::xpu::XPUStream(stream).queue();

  auto ctx = ccl::create_context(q.get_context());
  ccl::vector_class<ccl::pair_class<int, ccl::device>> devs_rank;
  devs_rank.emplace_back(rank, ccl::create_device(q.get_device()));

  auto xccl_kvs = get_kvs(rank_, *store_);
  auto comms = ccl::create_communicators(numRanks, devs_rank, ctx, xccl_kvs);
  std::shared_ptr<xcclComm_t> XCCLComm =
      std::make_shared<xcclComm_t>(std::move(comms[0]));

  std::lock_guard<std::mutex> lock(mutex_);
  devXCCLCommMap_.emplace(deviceKey, XCCLComm);
  xcclStreamsMap_.emplace(deviceKey, std::move(stream));
  xcclEventsMap_.emplace(deviceKey, at::xpu::XPUEvent());

  return XCCLComm;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupXCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType) {
  auto device = inputs[0].device();
  const auto key = std::to_string(device.index());
  auto comm = getXCCLComm(key, device);

  auto stream = xcclStreamsMap_.at(key);
  syncStream(device, xcclEventsMap_[key], stream);

  c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work;
  work = initWork(device, rank_, opType);
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  at::xpu::OptionalXPUGuard gpuGuard(device);
  pre(stream, work);
  for (const auto i : c10::irange(inputs.size())) {
    c10::xpu::XPUCachingAllocator::recordStream(
        inputs[i].storage().data_ptr(), stream);
    fn(inputs[i], outputs[i], *comm, stream);
  }
  post(stream, work);

  work->xcclEndEvent_->record(stream);
  std::vector<c10::Stream> streams = {stream.unwrap()};
  c10::MultiStreamGuard streamGuard(streams);
  std::vector<at::Device> devices{device};
  work->future_ = c10::make_intrusive<at::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(at::IValue(*work->outputs_));
  work->blockingWait_ = blockingWait_;

  return work;
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1, "Expecting one tensor only but got multiple");
  auto tensor = tensors.back();
  checkXPUTensor(tensor);

  RECORD_PARAM_COMMS_DATA(
      // static_cast<int>(
      //     this->getSequenceNumberForGroup() + 1), // seq + 1 to match
      //     collective
      1,
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      0, // globalRankStart
      1, // globalRankStride
      this->getSize()); // worldSize

  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        auto ccl_stream = ccl::create_stream(stream.queue());
        ccl::allreduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            ccl_stream);
        return;
      },
      OpType::ALLREDUCE);
}

} // namespace c10d

#endif // USE_C10D_XCCL
