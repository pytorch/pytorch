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

#include <c10/core/DeviceType.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/torch.h>

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
    {at::kShort, ccl::datatype::int16},
    {at::kInt, ccl::datatype::int32},
    {at::kLong, ccl::datatype::int64},
    {at::kHalf, ccl::datatype::float16},
    {at::kFloat, ccl::datatype::float32},
    {at::kDouble, ccl::datatype::float64},
    {at::kBFloat16, ccl::datatype::bfloat16},
    {at::kBool, ccl::datatype::uint8},
};

void check_gpu_single_tensor(const at::Tensor& tensor) {
  if (!tensor.is_xpu() || tensor.is_sparse()) {
    C10_THROW_ERROR(ValueError, "Tensors must be XPU and dense");
  }
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
  }
}
} // namespace

ccl::datatype getXcclDataType(at::ScalarType type) {
  auto it = xcclDatatypes.find(type);
  TORCH_CHECK_WITH(
      TypeError,
      it != xcclDatatypes.end(),
      "Input tensor data type is not supported for XCCL process group: ",
      type);
  return it->second;
}

} // namespace c10d

namespace {

static std::mutex xcclCommDevIdxMapMutex;
static std::unordered_map<std::shared_ptr<XCCLComm>, int> xcclCommDevIdxMap;

template <
    template <typename, typename, typename, typename, typename>
    class WorkXCCL,
    typename RunF,
    typename CommType,
    typename InputType,
    typename OutputType,
    typename attr_t>
c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> make_work_ccl(
    const std::vector<InputType>& inputs,
    const std::vector<OutputType>& outputs,
    RunF f,
    CommType& comms,
    attr_t& attr,
    int rank,
    c10d::OpType op_type) {
  c10::intrusive_ptr<WorkCCL<RunF, CommType, InputType, OutputType, attr_t>>
      ret_ptr = c10::make_intrusive<
          WorkCCL<RunF, CommType, InputType, OutputType, attr_t>>(
          inputs, outputs, f, comms, attr, rank, op_type);
  return ret_ptr;
}

ProcessGroupXCCL::WorkXCCL::WorkXCCL(
    std::vector<std::vector<at::Tensor>> outputTensors,
    int rank,
    c10d::OpType opType,
    const c10::optional<std::vector<at::Tensor>>& inputTensors)
    : Work(rank, opType, nullptr, inputTensors),
      outputTensors_(std::move(outputTensors)),
      future_(createFutureAsOutput(outputTensors)) {}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupXCCL::WorkXCCL::
    getFuture() {
  return future_;
}

c10::intrusive_ptr<Backend> ProcessGroupXCCL::createProcessGroupXCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size) {
  return c10::make_intrusive<ProcessGroupXCCL>(store, rank, size);
}

ProcessGroupXCCL::~ProcessGroupXCCL() {}

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
  xcclComm = ccl::create_communicator(numRanks, devs_rank, ctx, kvs);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    inInitializationCommMap_.emplace(deviceKey, xcclComm);
  }

  auto it = inInitializationCommMap_.find(deviceKey);
  if (it != inInitializationCommMap_.end()) {
    devXCCLCommMap_.emplace(deviceKey, std::move(it->second));
    inInitializationCommMap_.erase(deviceKey);

    xcclCommDevIdxMapMutex.lock();
    xcclCommDevIdxMap.emplace(xcclComm, device.index());
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
  auto xcclComm = getXCCLComm(key, device);

  std::vector<at::Tensor> inputs{input};
  std::vector<at::Tensor> outputs{output};

  c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work;
  // work =
  //     initWork(device, rank_, opType, profilingTitle, inputs, outputs,
  //     enqueue);

  work = make_work_ccl<WorkXCCL>(
      inputs, outputs, fn, xcclComm, attr, rank_, op_type);
  // pre(ncclStream, work);
  // ncclComm_t comm = ncclComm->getNcclComm();
  // post(ncclStream, work);

  return work;
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupNCCL::collective(
    at::Tensor& input,
    at::Tensor& output,
    Fn fn,
    OpType opType) {
  return collective<Fn>(
      input,
      output,
      fn,
      [](std::vector<ccl::stream>&) {},
      [](std::vector<ccl::stream>&) {},
      opType);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1, "Expecting one tensor only but got multiple");
  auto tensor = tensors.back();
  check_gpu_single_tensor(tensor);
  if (opts.reduceOp == ReduceOp::AVG) {
    TORCH_CHECK(false, "Cannot use ReduceOp AVG with XPU")
  }
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ccl::allreduce_attr attr,
          xcclComm_t comm,
          ccl::stream& stream) {
        ccl::event ret_evt;
        ccl::datatype datatype = getXcclDataType(input.scalar_type());
        ret_evt = ccl::allreduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            getXcclDataType(input.scalar_type()),
            xcclOp.at(opts.reduceOp),
            comm,
            stream,
            attr);
        return ret_evt;
      },
      OpType::ALLREDUCE);
}

} // namespace

#endif // USE_C10D_XCCL