#ifdef USE_C10D_XCCL

#include <torch/csrc/distributed/c10d/ProcessGroupXCCL.hpp>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <ATen/detail/FunctionTraits.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Optional.h>

namespace c10d {

namespace {
std::map<c10d::ReduceOp, ccl::reduction> xcclOps = {
    {ReduceOp::MIN, ccl::reduction::min},
    {ReduceOp::MAX, ccl::reduction::max},
    {ReduceOp::SUM, ccl::reduction::sum},
    {ReduceOp::PRODUCT, ccl::reduction::prod},
};

std::map<at::ScalarType, ccl::datatype> xcclDatatypes = {
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

void check_xpu_single_tensor(const at::Tensor& tensor) {
  if (!tensor.is_xpu() || tensor.is_sparse()) {
    C10_THROW_ERROR(ValueError, "Tensors must be XPU and dense");
  }
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
  }
}

ccl::datatype getXcclDataType(at::ScalarType type) {
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
    if (input.scalar_type() == at::kBool) {
      if (reduceOp == ReduceOp::SUM) {
        // For bool tensors, map sum to max, which both represent a bitwise or.
        // This is to prevent overflow issues with sum, since we use uint8 to
        // represent a bool (see xcclDatatypes mapping align with cuda).
        return ccl::reduction::max;
      }
    }
    return xcclOps.at(reduceOp);
  } catch (const std::out_of_range&) {
    C10_THROW_ERROR(
        ValueError,
        "Cannot use ReduceOp." + reduce_op_to_string(reduceOp) + " with XCCL");
  }
}

} // namespace

static std::mutex xcclCommDevIdxMapMutex;
static std::unordered_map<std::shared_ptr<xcclComm_t>, int> xcclCommDevIdxMap;
constexpr int64_t kSynchronizeBusyWaitMillis = 10;

ProcessGroupXCCL::WorkXCCL::WorkXCCL(
    at::Device& device,
    int rank,
    OpType opType,
    const std::optional<std::vector<at::Tensor>>& inputs)
    : Work(rank, opType, "profilingTitle", inputs),
      device_(device),
      workStartTime_(std::chrono::steady_clock::now()) {
  unsigned char enable_timing = 0;
  xcclEndEvent_ = std::make_shared<at::xpu::XPUEvent>(enable_timing);
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
            "Work ran for ",
            timeElapsed.count(),
            " milliseconds before timing out.");
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

  // Intel oneCCL requires passing CCL_LOCAL_RANK and CCL_LOCAL_SIZE for non-MPI
  // launchers.
  if (!with_mpirun()) {
    int local_rank = getXCCLEnvVar("LOCAL_RANK");
    int local_world_size = getXCCLEnvVar("LOCAL_WORLD_SIZE");
    if (local_rank == -1 || local_world_size == -1) {
      local_rank = rank;
      local_world_size = size;
    }
    setXCCLEnvVar("CCL_PROCESS_LAUNCHER", "none");
    setXCCLEnvVar("CCL_LOCAL_RANK", local_rank);
    setXCCLEnvVar("CCL_LOCAL_SIZE", local_world_size);
  }
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
  if (deviceKey.empty()) {
    C10_THROW_ERROR(
        DistBackendError,
        "Not able to create/get the XCCL Communicator since "
        "the devices are empty ");
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (devXCCLCommMap_.find(deviceKey) != devXCCLCommMap_.end()) {
      return devXCCLCommMap_[deviceKey];
    }
  }

  std::shared_ptr<xcclComm_t> XCCLComm;

  XCCL_KVS kvs = get_kvs(rank_, *store_);

  int numRanks, rank;
  numRanks = getSize();
  rank = getRank();

  c10::impl::VirtualGuardImpl impl(device.type());
  c10::Stream stream = impl.getStream(device);
  sycl::queue& q = c10::xpu::XPUStream(stream).queue();

  auto ctx = ccl::create_context(q.get_context());
  ccl::vector_class<ccl::pair_class<int, ccl::device>> devs_rank;
  devs_rank.emplace_back(rank, ccl::create_device(q.get_device()));

  auto comms = ccl::create_communicators(numRanks, devs_rank, ctx, kvs);
  XCCLComm = std::make_shared<xcclComm_t>(std::move(comms[0]));

  {
    std::lock_guard<std::mutex> lock(mutex_);
    inInitializationCommMap_.emplace(deviceKey, XCCLComm);
  }

  xcclStreams_.emplace(deviceKey, std::move(stream));

  auto it = inInitializationCommMap_.find(deviceKey);
  if (it != inInitializationCommMap_.end()) {
    devXCCLCommMap_.emplace(deviceKey, std::move(it->second));
    inInitializationCommMap_.erase(deviceKey);

    xcclCommDevIdxMapMutex.lock();
    xcclCommDevIdxMap.emplace(XCCLComm, device.index());
    xcclCommDevIdxMapMutex.unlock();
  }

  it = devXCCLCommMap_.find(deviceKey);
  TORCH_INTERNAL_ASSERT(
      it != devXCCLCommMap_.end(), "Communicators not populated in cache!");

  return it->second;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupXCCL::collective(
    at::Tensor& input,
    at::Tensor& output,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType) {
  using traits = function_traits<Fn>;
  using attr_t = typename traits::template arg<2>::type;
  attr_t attr = ccl::create_operation_attr<attr_t>();

  auto device = input.device();
  const auto key = std::to_string(device.index());
  auto comm = getXCCLComm(key, device);

  auto stream = xcclStreams_.at(key);
  std::vector<at::Tensor> outputs{output};

  c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work;

  work = initWork(device, rank_, opType);

  work->outputs_ =
      std::make_shared<std::vector<at::Tensor>>(std::move(outputs));
  c10::xpu::XPUCachingAllocator::recordStream(
      input.storage().data_ptr(), stream);

  auto ccl_stream = ccl::create_stream(stream.queue());

  fn(input, output, attr, *comm, ccl_stream);

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

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupXCCL::collective(
    at::Tensor& input,
    at::Tensor& output,
    Fn fn,
    OpType opType) {
  return collective<Fn>(
      input,
      output,
      fn,
      [](at::xpu::XPUStream&,
         c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work) {},
      [](at::xpu::XPUStream&,
         c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work) {},
      opType);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1, "Expecting one tensor only but got multiple");
  auto tensor = tensors.back();
  check_xpu_single_tensor(tensor);
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ccl::allreduce_attr attr,
          xcclComm_t& comm,
          ccl::stream& stream) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::event ret_evt;
        ret_evt = ccl::allreduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            stream,
            attr);
        return ret_evt;
      },
      OpType::ALLREDUCE);
}

} // namespace c10d

#endif // USE_C10D_XCCL
