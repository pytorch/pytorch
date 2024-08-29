#include <ATen/core/functional.h>
#include <torch/csrc/cuda/device_set.h>
#include <torch/csrc/cuda/nccl.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/hash.h>
#include <c10/util/irange.h>

#include <nccl.h>

#include <limits>
#include <sstream>
#include <type_traits>
#include <unordered_map>


xcclComm_t* to_xccl_comm(torch::xpu::xccl::xcclComm_t* var) {
  return reinterpret_cast<xcclComm_t*>(var);
}

xcclComm_t to_xccl_comm(torch::xpu::xccl::xcclComm_t var) {
  return reinterpret_cast<xcclComm_t>(var);
}


xcclDataType_t to_nccl_data_type(c10::ScalarType type) {
  switch (type) {
    case at::kFloat:
      return ccl::datatype::float32;
    case at::kHalf:
      return ccl::datatype::float16;
    case at::kDouble:
      return ccl::datatype::float64;
    case at::kLong:
      return ccl::datatype::int64;
    case at::kInt:
      return ccl::datatype::int32;
    case at::kChar:
      return ccl::datatype::int8;
    case at::kByte:
      return ccl::datatype::uint8;
    case at::kBool:
      return ccl::datatype::uint8;
    case at::kBFloat16:
      return ccl::datatype::bfloat16;
    default:
      TORCH_CHECK(false, "Unconvertible XCCL type ", type);
  }
}

ncclDataType_t to_xccl_data_type(const at::Tensor& t) {
  if (!t.is_xpu()) {
    TORCH_CHECK(
        false,
        "XCCL only supports XPU tensors, but got a tensor on ",
        t.device());
  }
  return to_xccl_data_type(t.scalar_type());
}

ccl::reduction to_xccl_red_op(int var) {
  return (ccl::reduction)(var);
}

namespace torch::xpu::xccl {

XCCL_KVS get_kvs(int rank, c10d::Store& store) {
  if (kvs)
    return kvs;
  // Each process group is with different store, so we use the unique key for
  // broadcast the bootstrap network information.
  std::string storeKey = "ccl_kvs";

  // Rank 0 broadcast the bootstrap network information to other ranks
  if (rank == 0) {
    kvs = ccl::create_main_kvs();
    ccl::kvs::address_type main_addr = kvs->get_address();
    auto ccl_kvs_addr = std::vector<uint8_t>(main_addr.begin(), main_addr.end());
    store.set(storeKey, ccl_kvs_addr);
  }
  else {
    auto ccl_kvs_addr = store.get(storeKey);
    if (ccl_kvs_addr.size() != ccl::kvs::address_max_size) {
      throw std::runtime_error(
              "Unexpected ccl kvs addr from the store\n");
    }
    ccl::kvs::address_type main_addr;
    std::copy_n(std::make_move_iterator(ccl_kvs_addr.begin()),
                ccl::kvs::address_max_size,
                main_addr.begin());
    kvs = ccl::create_kvs(main_addr);
  }

  return kvs;
}


using namespace at;

namespace detail {

void xcclCommInitAll(xcclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  for(int i = 0; i < nranks; i++) {
    newcomm[i] = ccl::create_communicator(nranks, i, get_kvs_addr)
  }
  c10::Stream dpcpp_stream = impl.getStream(devices[0]);
  ccl::vector_class<ccl::pair_class<int, ccl::device>> devs_rank;
  newcomm = ccl::create_communicators(nranks, devs_rank, ctx, )
}

struct XcclCommList {
  std::unique_ptr<xcclComm_t[]> comms;
  int ndevices;
  XcclCommList(const std::vector<int>& devices)
      : comms(new xcclComm_t[devices.size()]), ndevices(devices.size()) {
    xcclCommInitAll(
        to_xccl_comm(comms.get()), devices.size(), devices.data());
  }
  NcclCommList(NcclCommList&& foo) = default;
  ~NcclCommList() {
    if (comms) {
      for (const auto i : c10::irange(ndevices)) {
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
std::unordered_map<device_list, std::shared_ptr<Comms>> _communicators;
static std::unordered_map<device_list, NcclCommList, c10::hash<device_list>>
    _communicators;

ArrayRef<xcclComm_t> get_communicators(TensorList inputs) {
  static auto get_device = [](const at::Tensor& t) -> int {
    return t.get_device();
  };
  device_list devices = fmap(inputs, get_device);
  auto it = _communicators.find(devices);
  if (it == _communicators.end()) {
    it = _communicators.emplace(devices, devices).first;
  }
  return it->second;
}

static inline void check_tensor(
    const at::Tensor& input,
    const std::optional<at::Tensor>& output,
    int input_multiplier,
    int output_multiplier,
    int64_t ref_numel,
    ScalarType ref_dtype) {
  auto check_one = [&](const at::Tensor& tensor) {
    if (!tensor.is_xpu() || tensor.is_sparse()) {
      throw std::runtime_error(
          "input and output elements have to be xpu dense Tensors");
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

  for (const auto i : c10::irange(len)) {
    auto input = inputs[i];
    auto output = outputs[i];

    check_tensor(
        input, output, input_multiplier, output_multiplier, numel, dtype);

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
  auto len = inputs.size();

  if (len <= 0) {
    throw std::runtime_error("input sequence can't be empty");
  }

  device_set devices;
  int64_t numel = inputs[0].numel();
  auto dtype = inputs[0].scalar_type();

  for (const auto i : c10::irange(len)) {
    auto input = inputs[i];

    check_tensor(
        input,
        i == static_cast<std::remove_cv_t<decltype(i)>>(root)
            ? std::optional<at::Tensor>{output}
            : std::nullopt,
        input_multiplier,
        output_multiplier,
        numel,
        dtype);

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
#ifdef USE_XCCL
  device_set devices;
  for (auto& tensor : tensors) {
    if (!tensor.is_xpu() || tensor.is_sparse())
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
  constexpr std::uint64_t ver = (((uint64_t)NCCL_MAJOR) << 32) |
      (((uint64_t)NCCL_MINOR) << 16) | ((uint64_t)NCCL_PATCH);
  return ver;
#elif defined(USE_NCCL)
  // return major version "1"
  return ((uint64_t)1) << 32;
#else
  return 0;
#endif
}

ncclComm_t comm_init_rank(int nranks, const ncclUniqueId& comm_id, int rank) {
#ifdef USE_XCCL
  using namespace torch::xpu::xccl::detail;
  xcclComm_t comm;
  ncclUniqueId id = comm_id;
  NCCL_CHECK(ncclCommInitRank(
      to_nccl_comm(&comm), nranks, *(to_nccl_unique_id(&id)), rank));
  return comm;
#else
  return nullptr;
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

// Since NCCL 2.12.10, NCCL supports send/recv 0 byte:
// https://github.com/NVIDIA/nccl/issues/696. The issue of skipping send/recv
// is that it can cause deadlock when a rank send and recv 0 bytes so it's
// completely skipping the collective, causing mismatch across ranks
#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR > 13)))
template <typename T>
constexpr bool _nccl_should_send_recv(C10_UNUSED T _unused_) {
  return true;
}
#else
// old NCCL uses 0 byte message for synchronization
// Avoid send/recv when message size is zero
template <typename T>
inline bool _nccl_should_send_recv(T value) {
  return value != 0;
}
#endif
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
    auto device = tensors[i].get_device();
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
        tensors[i].data_ptr(),
        numel,
        data_type,
        0,
        to_nccl_comm(comm),
        stream));
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
  for (const auto i : c10::irange(len)) {
    auto device = inputs[i].device().index();
    device_guard.set_index(device);
    // Default to the current stream
    const auto stream = (streams.empty() || !streams[i])
        ? at::cuda::getCurrentCUDAStream(device).stream()
        : streams[i]->stream();

    ncclComm_t comm = comms_ref[i];
    NCCL_CHECK(ncclReduce(
        inputs[i].data_ptr(),
        static_cast<std::remove_cv_t<decltype(i)>>(root) == i
            ? output.data_ptr()
            : nullptr,
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
  for (const auto i : c10::irange(len)) {
    auto device = inputs[i].device().index();
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
  for (const auto i : c10::irange(len)) {
    auto device = inputs[i].device().index();
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
  for (const auto i : c10::irange(len)) {
    auto device = inputs[i].device().index();
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

void all2all_single_equal_split(
    at::Tensor& input,
    at::Tensor& output,
    int size,
    ncclComm_t _comm,
    at::cuda::CUDAStream& stream) {
#ifdef USE_NCCL
#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 7)))
  using namespace torch::cuda::nccl::detail;

  int numranks;
  auto type = to_nccl_data_type(input);
  size_t count = input.numel() / size;
  size_t rankdiff = input.nbytes() / size;
  const auto* sendbuff = reinterpret_cast<const char*>(input.const_data_ptr());
  auto* recvbuff = reinterpret_cast<char*>(output.data_ptr());
  auto comm = to_nccl_comm(_comm);
#if defined(USE_ROCM)
  NCCL_CHECK(ncclAllToAll(sendbuff, recvbuff, count, type, comm, stream));
#else
  NCCL_CHECK(ncclCommCount(comm, &numranks));
  NCCL_CHECK(ncclGroupStart());
  for (const auto r : c10::irange(numranks)) {
    if (_nccl_should_send_recv(count)) {
      NCCL_CHECK(
          ncclSend(sendbuff + r * rankdiff, count, type, r, comm, stream));
      NCCL_CHECK(
          ncclRecv(recvbuff + r * rankdiff, count, type, r, comm, stream));
    }
  }
#ifndef NCCL_HAS_COMM_NONBLOCKING
  NCCL_CHECK(ncclGroupEnd());
#else
  NCCL_CHECK_TIMEOUT(ncclGroupEnd(), _comm);
#endif
#endif
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
#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 7)))
  using namespace torch::cuda::nccl::detail;

  auto type = to_nccl_data_type(_type);
  auto comm = to_nccl_comm(_comm);
  int numranks;
  NCCL_CHECK(ncclCommCount(comm, &numranks));
  NCCL_CHECK(ncclGroupStart());
  for (const auto r : c10::irange(numranks)) {
    if (_nccl_should_send_recv(sendcounts[r])) {
      NCCL_CHECK(ncclSend(
          ((char*)sendbuff) + senddispls[r] * size,
          sendcounts[r],
          type,
          r,
          comm,
          stream));
    }
    if (_nccl_should_send_recv(recvcounts[r])) {
      NCCL_CHECK(ncclRecv(
          ((char*)recvbuff) + recvdispls[r] * size,
          recvcounts[r],
          type,
          r,
          comm,
          stream));
    }
  }
#ifndef NCCL_HAS_COMM_NONBLOCKING
  NCCL_CHECK(ncclGroupEnd());
#else
  NCCL_CHECK_TIMEOUT(ncclGroupEnd(), _comm);
#endif
#else
  AT_ERROR("all2all is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void all2all(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    ncclComm_t _comm,
    at::cuda::CUDAStream& stream) {
#ifdef USE_NCCL
#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 7)))
  using namespace torch::cuda::nccl::detail;
  auto comm = to_nccl_comm(_comm);

  NCCL_CHECK(ncclGroupStart());
  for (const auto r : c10::irange(outputTensors.size())) {
    at::Tensor& input = inputTensors[r];
    at::Tensor& output = outputTensors[r];

    if (_nccl_should_send_recv(input.numel())) {
      NCCL_CHECK(ncclSend(
          input.data_ptr(),
          input.numel(),
          to_nccl_data_type(input),
          r,
          comm,
          stream.stream()));
    }
    if (_nccl_should_send_recv(output.numel())) {
      NCCL_CHECK(ncclRecv(
          output.data_ptr(),
          output.numel(),
          to_nccl_data_type(output),
          r,
          comm,
          stream.stream()));
    }
  }
#ifndef NCCL_HAS_COMM_NONBLOCKING
  NCCL_CHECK(ncclGroupEnd());
#else
  NCCL_CHECK_TIMEOUT(ncclGroupEnd(), _comm);
#endif
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
#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 7)))
  using namespace torch::cuda::nccl::detail;
#ifndef NCCL_HAS_COMM_NONBLOCKING
  NCCL_CHECK(ncclSend(
      input.data_ptr(),
      input.numel(),
      to_nccl_data_type(input),
      dst,
      to_nccl_comm(comm),
      stream.stream()));
#else
  NCCL_CHECK_TIMEOUT(
      ncclSend(
          input.data_ptr(),
          input.numel(),
          to_nccl_data_type(input),
          dst,
          to_nccl_comm(comm),
          stream.stream()),
      comm);
#endif
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
#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 7)))
  using namespace torch::cuda::nccl::detail;
#ifndef NCCL_HAS_COMM_NONBLOCKING
  NCCL_CHECK(ncclRecv(
      output.data_ptr(),
      output.numel(),
      to_nccl_data_type(output),
      src,
      to_nccl_comm(comm),
      stream.stream()));
#else
  NCCL_CHECK_TIMEOUT(
      ncclRecv(
          output.data_ptr(),
          output.numel(),
          to_nccl_data_type(output),
          src,
          to_nccl_comm(comm),
          stream.stream()),
      comm);
#endif
#else
  AT_ERROR("Recv is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void gather(
    const at::Tensor& inputs,
    std::vector<at::Tensor>& outputs,
    ncclComm_t _comm,
    at::cuda::CUDAStream& stream,
    int32_t root) {
#ifdef USE_NCCL
#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 7)))
  using namespace torch::cuda::nccl::detail;

  auto comm = to_nccl_comm(_comm);
  int numranks, cur_rank;
  NCCL_CHECK(ncclCommCount(comm, &numranks));
  NCCL_CHECK(ncclCommUserRank(comm, &cur_rank));

  size_t count = inputs.numel();
  auto type = to_nccl_data_type(inputs);
  const auto* sendbuff = reinterpret_cast<const char*>(inputs.const_data_ptr());

  NCCL_CHECK(ncclGroupStart());

  if (cur_rank == root) {
    for (const auto r : c10::irange(numranks)) {
      if (r != root) {
        auto* recvbuff = reinterpret_cast<char*>(outputs[r].data_ptr());
        NCCL_CHECK(ncclRecv(recvbuff, count, type, r, comm, stream));
      } else {
        // on its own rank, simply copy from the input
        outputs[r].copy_(inputs);
      }
    }
  } else {
    NCCL_CHECK(ncclSend(sendbuff, count, type, root, comm, stream));
  }
#ifndef NCCL_HAS_COMM_NONBLOCKING
  NCCL_CHECK(ncclGroupEnd());
#else
  NCCL_CHECK_TIMEOUT(ncclGroupEnd(), _comm);
#endif

#else
  AT_ERROR("gather is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void scatter(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    ncclComm_t _comm,
    at::cuda::CUDAStream& stream,
    int32_t root) {
#ifdef USE_NCCL
#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR >= 7)))
  using namespace torch::cuda::nccl::detail;

  auto comm = to_nccl_comm(_comm);
  int numranks, cur_rank;
#ifndef NCCL_HAS_COMM_NONBLOCKING
  NCCL_CHECK(ncclCommCount(comm, &numranks));
  NCCL_CHECK(ncclCommUserRank(comm, &cur_rank));
#else
  NCCL_CHECK_TIMEOUT(ncclCommCount(comm, &numranks), _comm);
  NCCL_CHECK_TIMEOUT(ncclCommUserRank(comm, &cur_rank), _comm);
#endif
  NCCL_CHECK(ncclGroupStart());
  if (cur_rank == root) {
    for (const auto r : c10::irange(numranks)) {
      if (r != root) {
        size_t send_count = inputs[r].numel();
        auto send_type = to_nccl_data_type(inputs[r]);
        const auto* sendbuff =
            reinterpret_cast<const char*>(inputs[r].const_data_ptr());
        NCCL_CHECK(ncclSend(sendbuff, send_count, send_type, r, comm, stream));
      } else {
        // on its own rank, simply copy it to the output
        outputs.copy_(inputs[r]);
      }
    }
  } else {
    size_t recv_count = outputs.numel();
    auto recv_type = to_nccl_data_type(outputs);
    auto* recvbuff = reinterpret_cast<char*>(outputs.data_ptr());
    NCCL_CHECK(ncclRecv(recvbuff, recv_count, recv_type, root, comm, stream));
  }
#ifndef NCCL_HAS_COMM_NONBLOCKING
  NCCL_CHECK(ncclGroupEnd());
#else
  NCCL_CHECK_TIMEOUT(ncclGroupEnd(), _comm);
#endif
#else
  AT_ERROR("scatter is only supported for NCCL lib version >= 2.7.0");
#endif
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

} // namespace torch::cuda::nccl

