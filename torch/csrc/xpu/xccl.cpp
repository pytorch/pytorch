#include <ATen/core/functional.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/util/hash.h>
#include <c10/util/irange.h>

#include <torch/csrc/xpu/xccl.h>

#include <limits>
#include <sstream>
#include <type_traits>
#include <unordered_map>


ccl::datatype to_xccl_data_type(c10::ScalarType type) {
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

ccl::datatype to_xccl_data_type(const at::Tensor& t) {
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

XCCL_KVS kvs;
std::mutex kvs_mutex;

XCCL_KVS get_kvs(int rank, c10d::Store& store) {
  std::lock_guard<std::mutex> lock(kvs_mutex);
  if (kvs)
    return kvs;
  std::string storeKey = "ccl_kvs";

  // Rank 0 broadcast the bootstrap network information to other ranks
  if (rank == 0) {
    kvs = ccl::create_main_kvs();
    ccl::kvs::address_type main_addr = kvs->get_address();
    auto ccl_kvs_addr =
        std::vector<uint8_t>(main_addr.begin(), main_addr.end());
    store.set(storeKey, ccl_kvs_addr);
  } else {
    auto ccl_kvs_addr = store.get(storeKey);
    if (ccl_kvs_addr.size() != ccl::kvs::address_max_size) {
      throw std::runtime_error("Unexpected ccl kvs addr from the store\n");
    }
    ccl::kvs::address_type main_addr;
    std::copy_n(
      ccl_kvs_addr.begin(),
      ccl::kvs::address_max_size,
      main_addr.begin());
    kvs = ccl::create_kvs(main_addr);
  }

  return kvs;
}

using namespace at;

namespace detail {

// void xcclCommInitAll(xcclComm_t* newcomm, int nranks, ncclUniqueId commId,
// int myrank) {
//   for(int i = 0; i < nranks; i++) {
//     newcomm[i] = ccl::create_communicator(nranks, i, get_kvs_addr)
//   }
//   c10::Stream dpcpp_stream = impl.getStream(devices[0]);
//   ccl::vector_class<ccl::pair_class<int, ccl::device>> devs_rank;
//   newcomm = ccl::create_communicators(nranks, devs_rank, ctx, )
// }

// struct XcclCommList {
//   std::unique_ptr<xcclComm_t[]> comms;
//   int ndevices;
//   XcclCommList(const std::vector<int>& devices)
//       : comms(new xcclComm_t[devices.size()]), ndevices(devices.size()) {
//     xcclCommInitAll(
//         to_xccl_comm(comms.get()), devices.size(), devices.data());
//   }
//   NcclCommList(NcclCommList&& foo) = default;
//   ~NcclCommList() {
//     if (comms) {
//       for (const auto i : c10::irange(ndevices)) {
//         comm_destroy(comms[i]);
//       }
//     }
//   }
//   ArrayRef<ncclComm_t> ref() const {
//     return ArrayRef<ncclComm_t>(comms.get(), ndevices);
//   }
// };

// using device_list = std::vector<int>;
// // accesses to this object have to be guarded by THC's CudaFreeMutex
// std::unordered_map<device_list, std::shared_ptr<Comms>> _communicators;
// static std::unordered_map<device_list, NcclCommList, c10::hash<device_list>>
//     _communicators;

// ArrayRef<xcclComm_t> get_communicators(TensorList inputs) {
//   static auto get_device = [](const at::Tensor& t) -> int {
//     return t.get_device();
//   };
//   device_list devices = fmap(inputs, get_device);
//   auto it = _communicators.find(devices);
//   if (it == _communicators.end()) {
//     it = _communicators.emplace(devices, devices).first;
//   }
//   return it->second;
// }

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

// void check_inputs(
//     TensorList inputs,
//     TensorList outputs,
//     int input_multiplier,
//     int output_multiplier) {
//   // len(inputs) == len(outputs)
//   size_t len = inputs.size();

//   if (len <= 0) {
//     throw std::runtime_error("input sequence can't be empty");
//   }

//   if (len != outputs.size()) {
//     std::stringstream err;
//     err << "inputs and outputs sequences have to be of the same length, but got input of length "
//         << len << " and output of length " << outputs.size();
//     throw std::runtime_error(err.str());
//   }

//   device_set devices;
//   int64_t numel = inputs[0].numel();
//   auto dtype = inputs[0].scalar_type();

//   for (const auto i : c10::irange(len)) {
//     auto input = inputs[i];
//     auto output = outputs[i];

//     check_tensor(
//         input, output, input_multiplier, output_multiplier, numel, dtype);

//     auto input_device = input.get_device();
//     // inputs must be on unique devices
//     if (devices.test(input_device)) {
//       throw std::runtime_error("inputs must be on unique devices");
//     }
//     devices.set(input_device);
//   }
// }

// void check_inputs(
//     TensorList inputs,
//     const at::Tensor& output,
//     int root,
//     int input_multiplier,
//     int output_multiplier) {
//   auto len = inputs.size();

//   if (len <= 0) {
//     throw std::runtime_error("input sequence can't be empty");
//   }

//   device_set devices;
//   int64_t numel = inputs[0].numel();
//   auto dtype = inputs[0].scalar_type();

//   for (const auto i : c10::irange(len)) {
//     auto input = inputs[i];

//     check_tensor(
//         input,
//         i == static_cast<std::remove_cv_t<decltype(i)>>(root)
//             ? std::optional<at::Tensor>{output}
//             : std::nullopt,
//         input_multiplier,
//         output_multiplier,
//         numel,
//         dtype);

//     auto input_device = input.get_device();
//     // inputs must be on unique devices
//     if (devices.test(input_device)) {
//       throw std::runtime_error("inputs must be on unique devices");
//     }
//     devices.set(input_device);
//   }
// }

} // namespace detail

// std::uint64_t version() {
// #if defined(NCCL_MAJOR)
//   constexpr std::uint64_t ver = (((uint64_t)NCCL_MAJOR) << 32) |
//       (((uint64_t)NCCL_MINOR) << 16) | ((uint64_t)NCCL_PATCH);
//   return ver;
// #elif defined(USE_NCCL)
//   // return major version "1"
//   return ((uint64_t)1) << 32;
// #else
//   return 0;
// #endif
// }

// ncclComm_t comm_init_rank(int nranks, const ncclUniqueId& comm_id, int rank)
// { #ifdef USE_XCCL
//   using namespace torch::xpu::xccl::detail;
//   xcclComm_t comm;
//   ncclUniqueId id = comm_id;
//   NCCL_CHECK(ncclCommInitRank(
//       to_nccl_comm(&comm), nranks, *(to_nccl_unique_id(&id)), rank));
//   return comm;
// #else
//   return nullptr;
// #endif
// }

// namespace {

//         ret_evt = torch::xpu::xccl::all_reduce(
//             input,
//             output,
//             datatype,
//             xcclOp.at(opts.reduceOp),
//             comm,
//             attr,
//             stream,
//             root);

// void all_reduce(
//     at::Tensor& input,
//     at::Tensor& output,
//     ccl::datatype datatype,
//     ccl::reduction op,
//     const stream_list& streams,
//     const comm_list& user_comms) {
// #ifdef USE_XCCL
//   using namespace torch::cuda::nccl::detail;
//   check_inputs(inputs, outputs, 1, 1);
//   const auto len = inputs.size();

//   auto data_type = to_nccl_data_type(inputs[0]);

//   const auto count = inputs[0].numel();
//   auto comms_ref = user_comms.empty() ? get_communicators(inputs)
//                                       : ArrayRef<ncclComm_t>(user_comms);

//   AutoNcclGroup nccl_group_guard;
//   at::cuda::OptionalCUDAGuard device_guard;
//   for (const auto i : c10::irange(len)) {
//     auto device = inputs[i].device().index();
//     device_guard.set_index(device);
//     // Default to the current stream
//     const auto stream = (streams.empty() || !streams[i])
//         ? at::cuda::getCurrentCUDAStream(device).stream()
//         : streams[i]->stream();

//     ncclComm_t comm = comms_ref[i];
//     NCCL_CHECK(ncclAllReduce(
//         inputs[i].data_ptr(),
//         outputs[i].data_ptr(),
//         count,
//         data_type,
//         to_nccl_red_op(op),
//         to_nccl_comm(comm),
//         stream));
//   }
// #else
//   AT_ERROR("PyTorch built without NCCL support");
// #endif
// }

} // namespace torch::xpu::xccl
