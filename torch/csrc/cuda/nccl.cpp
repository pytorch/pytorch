#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/cuda/device_set.h>
#include <ATen/core/functional.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/hash.h>

#include <THC/THC.h>

#include <nccl.h>

#include <limits>
#include <sstream>
#include <type_traits>
#include <unordered_map>


ncclComm_t* to_nccl_comm(torch::cuda::nccl::ncclComm_t* var) {
  return reinterpret_cast<ncclComm_t*>(var);
}

ncclComm_t to_nccl_comm(torch::cuda::nccl::ncclComm_t var) {
  return reinterpret_cast<ncclComm_t>(var);
}

ncclUniqueId* to_nccl_unique_id(torch::cuda::nccl::ncclUniqueId* var) {
  return reinterpret_cast<ncclUniqueId*>(var);
}

ncclResult_t to_nccl_result(torch::cuda::nccl::ncclResult var) {
  switch (var) {
    case torch::cuda::nccl::ncclResult::Success:
      return ncclResult_t::ncclSuccess;
    case torch::cuda::nccl::ncclResult::UnhandledCudaError:
      return ncclResult_t::ncclUnhandledCudaError;
    case torch::cuda::nccl::ncclResult::SystemError:
      return ncclResult_t::ncclSystemError;
    case torch::cuda::nccl::ncclResult::InternalError:
      return ncclResult_t::ncclInternalError;
    case torch::cuda::nccl::ncclResult::InvalidArgument:
      return ncclResult_t::ncclInvalidArgument;
    case torch::cuda::nccl::ncclResult::InvalidUsage:
      return ncclResult_t::ncclInvalidUsage;
    case torch::cuda::nccl::ncclResult::NumResults:
      return ncclResult_t::ncclNumResults;
    default:
      throw std::runtime_error("Unconvertible NCCL type");
  }
}

torch::cuda::nccl::ncclResult from_nccl_result(ncclResult_t var) {
    switch (var) {
    case ncclSuccess:
      return torch::cuda::nccl::ncclResult::Success;
    case ncclUnhandledCudaError:
      return torch::cuda::nccl::ncclResult::UnhandledCudaError;
    case ncclSystemError:
      return torch::cuda::nccl::ncclResult::SystemError;
    case ncclInternalError:
      return torch::cuda::nccl::ncclResult::InternalError;
    case ncclInvalidArgument:
      return torch::cuda::nccl::ncclResult::InvalidArgument;
    case ncclInvalidUsage:
      return torch::cuda::nccl::ncclResult::InvalidUsage;
    case ncclNumResults:
      return torch::cuda::nccl::ncclResult::NumResults;
    default:
      throw std::runtime_error("Unconvertible NCCL type");
  }
}

ncclDataType_t to_nccl_data_type(c10::ScalarType type) {
  switch (type) {
    case at::kFloat:
      return ncclDataType_t::ncclFloat;
    case at::kHalf:
      return ncclDataType_t::ncclHalf;
    case at::kDouble:
      return ncclDataType_t::ncclDouble;
    case at::kLong:
      return ncclDataType_t::ncclInt64;
    case at::kInt:
      return ncclDataType_t::ncclInt;
    case at::kChar:
      return ncclDataType_t::ncclChar;
    case at::kByte:
      return ncclDataType_t::ncclUint8;
    case at::kBool:
      return ncclDataType_t::ncclUint8;
#if defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 301
    case at::kBFloat16:
      return ncclDataType_t::ncclBfloat16;
#endif
    default:
      TORCH_CHECK(false, "Unconvertible NCCL type ", type);
  }
}

ncclDataType_t to_nccl_data_type(const at::Tensor& t) {
  if (!t.is_cuda()) {
    TORCH_CHECK(false, "NCCL only supports CUDA tensors, but got a tensor on ", t.device());
  }
  return to_nccl_data_type(t.scalar_type());
}

ncclRedOp_t to_nccl_red_op(int var) {
  return (ncclRedOp_t)(var);
}

namespace torch {
namespace cuda {
namespace nccl {

using namespace at;

namespace detail {

static inline void NCCL_CHECK(ncclResult_t result) {
  NCCL_CHECK(from_nccl_result(result));
}

struct AutoNcclGroup {
  AutoNcclGroup() {
    (c10::cuda::CUDACachingAllocator::getFreeMutex())->lock();
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
    NCCL_CHECK(ncclGroupStart());
#endif
  }
  ~AutoNcclGroup() {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
    NCCL_CHECK(ncclGroupEnd());
#endif
    (c10::cuda::CUDACachingAllocator::getFreeMutex())->unlock();
  }
};

void throw_nccl_error(torch::cuda::nccl::ncclResult status) {
  std::ostringstream err;
  err << "NCCL Error " << static_cast<int>(status) << ": " << ncclGetErrorString(to_nccl_result(status));
  throw std::runtime_error(err.str());
}

struct NcclCommList {
  std::unique_ptr<ncclComm_t[]> comms;
  int ndevices;
  NcclCommList(const std::vector<int>& devices)
      : comms(new ncclComm_t[devices.size()]), ndevices(devices.size()) {
    NCCL_CHECK(
      ncclCommInitAll(to_nccl_comm(comms.get()), devices.size(), devices.data()));
  }
  NcclCommList(NcclCommList&& foo) = default;
  ~NcclCommList() {
    if (comms) {
      for (int i = 0; i < ndevices; i++) {
        int dummy_var;
        if (cudaGetDevice(&dummy_var) != cudaSuccess) {
          /* there are cases when this destructor is called after the
           CUDA driver is already unloaded from the process.
           In these cases, skip ncclCommDestroy */
          return;
        }
        comm_destroy(comms[i]);
      }
    }
  }
  ArrayRef<ncclComm_t> ref() const {
    return ArrayRef<ncclComm_t>(comms.get(), ndevices);
  }
};

using device_list = std::vector<int>;
// accesses to this object have to be guarded by THC's CudaFreeMutex
static std::unordered_map<device_list, NcclCommList, c10::hash<device_list>>
    _communicators;

ArrayRef<ncclComm_t> get_communicators(TensorList inputs) {
  static auto get_device = [](const at::Tensor& t) -> int {
    return t.get_device();
  };
  device_list devices = fmap(inputs, get_device);
  auto it = _communicators.find(devices);
  if (it == _communicators.end())
    std::tie(it, std::ignore) = _communicators.emplace(devices, devices);
  return it->second.ref();
}

static inline
void check_tensor(
    const at::Tensor& input,
    const at::optional<at::Tensor>& output,
    int input_multiplier,
    int output_multiplier,
    int64_t ref_numel,
    ScalarType ref_dtype) {

  auto check_one = [&](const at::Tensor &tensor) {
    if (!tensor.is_cuda() || tensor.is_sparse()) {
      throw std::runtime_error(
          "input and output elements have to be cuda dense Tensors");
    }

    if (ref_dtype != tensor.scalar_type()) {
      throw std::runtime_error(
          "all inputs and outputs must be of the same Tensor dtype");
    }

    if (!tensor.is_contiguous()) {
      throw std::runtime_error("all inputs and outputs have to be contiguous");
    }
  };

  check_one(input);

  // all inputs must be same size
  if (input.numel() != ref_numel) {
    throw std::runtime_error(
        "all inputs must have the same number of elements");
  }

  if (output) {
    check_one(*output);

    // inputs and outputs must be on same device respectively
    if (input.get_device() != output->get_device()) {
      throw std::runtime_error("input and output must be on the same device");
    }

    if (output->numel() * output_multiplier != ref_numel * input_multiplier) {
      throw std::runtime_error(
          "output must be of size input_size * size_multiplier");
    }
  }
}

void check_inputs(
    TensorList inputs,
    TensorList outputs,
    int input_multiplier,
    int output_multiplier) {
  // len(inputs) == len(outputs)
  size_t len = inputs.size();

  if (len <= 0) {
    throw std::runtime_error("input sequence can't be empty");
  }

  if (len != outputs.size()) {
    std::stringstream err;
    err << "inputs and outputs sequences have to be of the same length, but got input of length "
        << len << " and output of length " << outputs.size();
    throw std::runtime_error(err.str());
  }

  device_set devices;
  int64_t numel = inputs[0].numel();
  auto dtype = inputs[0].scalar_type();

  for (size_t i = 0; i < len; i++) {
    auto input = inputs[i];
    auto output = outputs[i];

    check_tensor(input, output, input_multiplier, output_multiplier, numel, dtype);

    auto input_device = input.get_device();
    // inputs must be on unique devices
    if (devices.test(input_device)) {
      throw std::runtime_error("inputs must be on unique devices");
    }
    devices.set(input_device);
  }
}

void check_inputs(
    TensorList inputs,
    const at::Tensor& output,
    int root,
    int input_multiplier,
    int output_multiplier) {
  size_t len = inputs.size();

  if (len <= 0) {
    throw std::runtime_error("input sequence can't be empty");
  }

  device_set devices;
  int64_t numel = inputs[0].numel();
  auto dtype = inputs[0].scalar_type();

  for (size_t i = 0; i < len; i++) {
    auto input = inputs[i];

    check_tensor(
      input,
      i == root ? at::optional<at::Tensor>{output} : at::nullopt,
      input_multiplier, output_multiplier, numel, dtype);

    auto input_device = input.get_device();
    // inputs must be on unique devices
    if (devices.test(input_device)) {
      throw std::runtime_error("inputs must be on unique devices");
    }
    devices.set(input_device);
  }
}

} // namespace detail

bool is_available(TensorList tensors) {
#ifdef USE_NCCL
  device_set devices;
  for (auto& tensor : tensors) {
    if (!tensor.is_cuda() || tensor.is_sparse())
      return false;
    if (!tensor.is_contiguous())
      return false;
    auto device = tensor.get_device();
    if (devices[device])
      return false;
    devices[device] = true;
  }
  return true;
#else
  return false;
#endif
}

std::uint64_t version() {
#if defined(NCCL_MAJOR)
  return NCCL_MAJOR * 1000 + NCCL_MINOR * 100 + NCCL_PATCH;
#elif defined(USE_NCCL)
  return 1000;
#else
  return 0;
#endif
}

void get_unique_id(ncclUniqueId& id)
{
#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  NCCL_CHECK(ncclGetUniqueId(to_nccl_unique_id(&id)));
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

ncclComm_t comm_init_rank(int nranks, const ncclUniqueId& comm_id, int rank) {
#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  ncclComm_t comm;
  ncclUniqueId id = comm_id;
  NCCL_CHECK(ncclCommInitRank(
    to_nccl_comm(&comm),
    nranks,
    *(to_nccl_unique_id(&id)),
    rank));
  return comm;
#else
  return nullptr;
#endif
}

void comm_destroy(ncclComm_t comm)
{
  /*
   * TODO(T30279827) Temporarily disable calling ncclCommDestroy
   * Calling ncclCommDestroy while program exiting is undefined
   * according to Nvidia, and lead to segfault in NCCL 2
   * (whether it is called before or after the CUDA runtime destructor).
   * Temporarily disable it in destructor to avoid segfault.
   * Following up with Nvidia for long term solution.
   */
  return;

#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  NCCL_CHECK(ncclCommDestroy(to_nccl_comm(comm)));
#endif
}

namespace {
// NCCL changed the numerical type used for count between NCCL1 and NCCL2.
// So we use the following struct, which gets the type of the second argument
// of T, if T is a function type, with ncclBcast, to get that type statically
// and programmatically.

template <typename T>
struct GetSecondArgType;

template <typename R, typename Arg0, typename Arg1, typename... Args>
struct GetSecondArgType<R(Arg0, Arg1, Args...)> {
  typedef typename std::decay<Arg1>::type type;
};

constexpr auto count_max =
    std::numeric_limits<GetSecondArgType<decltype(ncclBcast)>::type>::max();
} // namespace

size_t get_max_count() {
  return count_max;
}

void broadcast(
    TensorList tensors,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  check_inputs(tensors, tensors, 1, 1);
  auto data_type = to_nccl_data_type(tensors[0]);
  int64_t numel = tensors[0].numel();

  const auto comms = user_comms.empty() ? get_communicators(tensors)
                                        : ArrayRef<ncclComm_t>(user_comms);

  AutoNcclGroup nccl_group_guard;
  at::cuda::OptionalCUDAGuard device_guard;
  for (size_t i = 0, num_tensors = tensors.size(); i < num_tensors; i++) {
    int device = tensors[i].get_device();
    device_guard.set_index(device);
    // Default to the current stream
    const auto stream = (streams.empty() || !streams[i])
        ? at::cuda::getCurrentCUDAStream(device).stream()
        : streams[i]->stream();
    TORCH_CHECK(
        static_cast<uint64_t>(numel) <= static_cast<uint64_t>(count_max),
        "Broadcast tensor has ",
        numel,
        " elements, which exceeds the "
        "maximum NCCL supports (",
        count_max,
        ")");
    ncclComm_t comm = comms[i];
    NCCL_CHECK(ncclBcast(
        tensors[i].data_ptr(), numel, data_type, 0, to_nccl_comm(comm), stream));
  }
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void reduce(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& output,
    int32_t root,
    int32_t op,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  TORCH_CHECK(
      root >= 0 && static_cast<size_t>(root) < inputs.size(), "invalid root");

  check_inputs(inputs, output, root, 1, 1);
  const auto len = inputs.size();

  auto data_type = to_nccl_data_type(inputs[0]);

  const auto count = inputs[0].numel();
  auto comms_ref = user_comms.empty() ? get_communicators(inputs)
                                      : ArrayRef<ncclComm_t>(user_comms);

  AutoNcclGroup nccl_group_guard;
  at::cuda::OptionalCUDAGuard device_guard;
  for (size_t i = 0; i < len; i++) {
    int device = inputs[i].device().index();
    device_guard.set_index(device);
    // Default to the current stream
    const auto stream = (streams.empty() || !streams[i])
        ? at::cuda::getCurrentCUDAStream(device).stream()
        : streams[i]->stream();

    ncclComm_t comm = comms_ref[i];
    NCCL_CHECK(ncclReduce(
        inputs[i].data_ptr(),
        root == i ? output.data_ptr() : nullptr,
        count,
        data_type,
        to_nccl_red_op(op),
        root,
        to_nccl_comm(comm),
        stream));
  }
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void reduce(
    std::vector<at::Tensor>& inputs,
    int32_t root,
    int32_t op,
    const stream_list& streams,
    const comm_list& user_comms) {
  reduce(inputs, /*output=*/inputs[root], root, op, streams, user_comms);
}

void all_reduce(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  check_inputs(inputs, outputs, 1, 1);
  const auto len = inputs.size();

  auto data_type = to_nccl_data_type(inputs[0]);

  const auto count = inputs[0].numel();
  auto comms_ref = user_comms.empty() ? get_communicators(inputs)
                                      : ArrayRef<ncclComm_t>(user_comms);

  AutoNcclGroup nccl_group_guard;
  at::cuda::OptionalCUDAGuard device_guard;
  for (size_t i = 0; i < len; i++) {
    int device = inputs[i].device().index();
    device_guard.set_index(device);
    // Default to the current stream
    const auto stream = (streams.empty() || !streams[i])
        ? at::cuda::getCurrentCUDAStream(device).stream()
        : streams[i]->stream();

    ncclComm_t comm = comms_ref[i];
    NCCL_CHECK(ncclAllReduce(
        inputs[i].data_ptr(),
        outputs[i].data_ptr(),
        count,
        data_type,
        to_nccl_red_op(op),
        to_nccl_comm(comm),
        stream));
  }
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void reduce_scatter(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  const auto len = inputs.size();
  check_inputs(inputs, outputs, 1, len);

  auto data_type = to_nccl_data_type(inputs[0]);

  const auto count = inputs[0].numel() / len;
  auto comms_ref = user_comms.empty() ? get_communicators(inputs)
                                      : ArrayRef<ncclComm_t>(user_comms);

  AutoNcclGroup nccl_group_guard;
  at::cuda::OptionalCUDAGuard device_guard;
  for (size_t i = 0; i < len; i++) {
    int device = inputs[i].device().index();
    device_guard.set_index(device);
    // Default to the current stream
    const auto stream = (streams.empty() || !streams[i])
        ? at::cuda::getCurrentCUDAStream(device).stream()
        : streams[i]->stream();

    ncclComm_t comm = comms_ref[i];
    NCCL_CHECK(ncclReduceScatter(
        inputs[i].data_ptr(),
        outputs[i].data_ptr(),
        count,
        data_type,
        to_nccl_red_op(op),
        to_nccl_comm(comm),
        stream));
  }
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void all_gather(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  const auto len = inputs.size();
  check_inputs(inputs, outputs, len, 1);

  auto data_type = to_nccl_data_type(inputs[0]);

  const auto count = inputs[0].numel();
  auto comms_ref = user_comms.empty() ? get_communicators(inputs)
                                      : ArrayRef<ncclComm_t>(user_comms);

  AutoNcclGroup nccl_group_guard;
  at::cuda::OptionalCUDAGuard device_guard;
  for (size_t i = 0; i < len; i++) {
    int device = inputs[i].device().index();
    device_guard.set_index(device);
    // Default to the current stream
    const auto stream = (streams.empty() || !streams[i])
        ? at::cuda::getCurrentCUDAStream(device).stream()
        : streams[i]->stream();

    ncclComm_t comm = comms_ref[i];
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
      NCCL_CHECK(ncclAllGather(
          inputs[i].data_ptr(),
          outputs[i].data_ptr(),
          count,
          data_type,
          to_nccl_comm(comm),
          stream));
#else
      NCCL_CHECK(ncclAllGather(
          inputs[i].data_ptr(),
          count,
          data_type,
          outputs[i].data_ptr(),
          to_nccl_comm(comm),
          stream));
#endif
  }
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void all2all_single_equal_split(at::Tensor& input,
             at::Tensor& output,
             int size,
             ncclComm_t _comm,
             at::cuda::CUDAStream& stream) {
#ifdef USE_NCCL
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && (NCCL_MAJOR * 10 + NCCL_MINOR) >= 27
  using namespace torch::cuda::nccl::detail;

  int numranks;
  auto type = to_nccl_data_type(input);
  size_t count = input.numel() / size;
  size_t rankdiff = input.nbytes() / size;
  const auto* sendbuff = reinterpret_cast<char*>(input.data_ptr());
  auto* recvbuff = reinterpret_cast<char *>(output.data_ptr());
  auto comm = to_nccl_comm(_comm);
  NCCL_CHECK(ncclCommCount(comm, &numranks));
  NCCL_CHECK(ncclGroupStart());
  for (int r = 0; r < numranks; r++) {
    // NCCL uses 0 byte message for synchronization
    // Avoid send/recv when message size is zero
    if (count != 0) {
      NCCL_CHECK(ncclSend(sendbuff + r * rankdiff, count, type, r, comm, stream));
      NCCL_CHECK(ncclRecv(recvbuff + r * rankdiff, count, type, r, comm, stream));
    }
  }
  NCCL_CHECK(ncclGroupEnd());
#else
  AT_ERROR("all2all is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void all2all_single_unequal_split(
    void* sendbuff,
    const size_t* sendcounts,
    const size_t* senddispls,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* recvdispls,
    size_t size,
    c10::ScalarType _type,
    ncclComm_t _comm,
    at::cuda::CUDAStream& stream) {
#ifdef USE_NCCL
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && (NCCL_MAJOR * 10 + NCCL_MINOR) >= 27
  using namespace torch::cuda::nccl::detail;

  auto type = to_nccl_data_type(_type);
  auto comm = to_nccl_comm(_comm);
  int numranks;
  NCCL_CHECK(ncclCommCount(comm, &numranks));
  NCCL_CHECK(ncclGroupStart());
  for (int r = 0; r < numranks; r++) {
    // NCCL uses 0 byte message for synchronization
    // Avoid send/recv when message size is zero
    if (sendcounts[r] != 0) {
      NCCL_CHECK(ncclSend(
          ((char*)sendbuff) + senddispls[r] * size,
          sendcounts[r],
          type,
          r,
          comm,
          stream));
    }
    if (recvcounts[r] != 0) {
      NCCL_CHECK(ncclRecv(
          ((char*)recvbuff) + recvdispls[r] * size,
          recvcounts[r],
          type,
          r,
          comm,
          stream));
    }
  }
  NCCL_CHECK(ncclGroupEnd());
#else
  AT_ERROR("all2all is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void all2all(std::vector<at::Tensor>& outputTensors,
             std::vector<at::Tensor>& inputTensors,
             ncclComm_t _comm,
             at::cuda::CUDAStream& stream) {
#ifdef USE_NCCL
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && (NCCL_MAJOR * 10 + NCCL_MINOR) >= 27
  using namespace torch::cuda::nccl::detail;
  auto comm = to_nccl_comm(_comm);

  NCCL_CHECK(ncclGroupStart());
  for (size_t r = 0; r < outputTensors.size(); r++) {
    at::Tensor &input = inputTensors[r];
    at::Tensor &output = outputTensors[r];
    if (input.numel() != 0) {
      NCCL_CHECK(ncclSend(
          input.data_ptr(),
          input.numel(),
          to_nccl_data_type(input),
          r,
          comm,
          stream.stream()));
    }
    if (output.numel() != 0) {
      NCCL_CHECK(ncclRecv(
          output.data_ptr(),
          output.numel(),
          to_nccl_data_type(output),
          r,
          comm,
          stream.stream()));
    }
  }
  NCCL_CHECK(ncclGroupEnd());
#else
  AT_ERROR("all2all is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void send(
    const at::Tensor& input,
    ncclComm_t comm,
    at::cuda::CUDAStream stream,
    int dst) {
#ifdef USE_NCCL
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 7)
  using namespace torch::cuda::nccl::detail;
  NCCL_CHECK(ncclSend(
      input.data_ptr(),
      input.numel(),
      to_nccl_data_type(input),
      dst,
      to_nccl_comm(comm),
      stream.stream()));
#else
  AT_ERROR("Send is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void recv(
    at::Tensor& output,
    ncclComm_t comm,
    at::cuda::CUDAStream stream,
    int src) {
#ifdef USE_NCCL
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 7)
  using namespace torch::cuda::nccl::detail;
  NCCL_CHECK(ncclRecv(
      output.data_ptr(),
      output.numel(),
      to_nccl_data_type(output),
      src,
      to_nccl_comm(comm),
      stream.stream()));
#else
  AT_ERROR("Recv is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

} // namespace nccl
} // namespace cuda
} // namespace torch
