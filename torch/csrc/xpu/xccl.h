#pragma once

#include <ATen/ATen.h>
#include <oneapi/ccl.hpp>
#include <cstddef>
#include <optional>
#include <vector>

namespace torch::xpu::xccl {

using xcclComm_t = ccl::communicator;

using XCCL_KVS = ccl::shared_ptr_class<ccl::kvs>;

ccl::shared_ptr_class<ccl::kvs> kvs;
std::vector<uint8_t> kvs_addr;

XCCL_KVS get_kvs(int rank, c10d::Store& store)
class Comms {
public:

  explicit Comms(ccl::vector_class<xcclComm_t> &comms) :
    comms(std::move(comms)), streams{} {}

  explicit Comms(ccl::vector_class<xcclComm_t> &comms, ccl::vector_class<ccl::stream> &streams, std::vector<c10::Stream> &torch_streams) :
    comms(std::move(comms)), streams(std::move(streams)), torch_streams(std::move(torch_streams)) {}

  ~Comms() noexcept(false) {}

  Comms() = delete;

  Comms(const Comms &) = delete;

  Comms &operator=(const Comms &) = delete;

  Comms(Comms &&other) : comms(std::move(other.comms)), streams(std::move(other.streams)),
                         torch_streams(std::move(other.torch_streams)) {}

  Comms &operator=(Comms &&other) {
    std::swap(comms, other.comms);
    std::swap(streams, other.streams);
    std::swap(torch_streams, other.torch_streams);
    return *this;
  }

public:
  // The Communicators used by XCCL
  ccl::vector_class<xcclComm_t> comms;
  // The streams used by XCCL
  ccl::vector_class<ccl::stream> streams;
  // one to one mapping the torch streams to the ccl::stream.
  std::vector<c10::Stream> torch_streams;
};

enum class xcclRedOp { Sum = 0, Prod = 1, Max = 2, Min = 3};

enum class xcclDataType {
  Int8 = 0,
  Char = 0,
  Uint8 = 1,
  Int32 = 2,
  Int = 2,
  Uint32 = 3,
  Int64 = 4,
  Uint64 = 5,
  Float16 = 6,
  Half = 6,
  Float32 = 7,
  Float = 7,
  Float64 = 8,
  Double = 8,
  Bfloat16 = 9,
  NumTypes = 10
};

namespace detail {

 at::ArrayRef<xcclComm_t> get_communicators(
    at::TensorList inputs);
 void check_inputs(
    at::TensorList inputs,
    at::TensorList outputs,
    int input_multiplier,
    int output_multiplier);
 void check_inputs(
    at::TensorList inputs,
    const at::Tensor& output,
    int root,
    int input_multiplier,
    int output_multiplier);

} // namespace detail

using comm_list = std::vector<xor>;
using stream_list = std::vector<std::optional<at::xpu::XPUStream>>;

 std::uint64_t version();
 const char* version_suffix();

bool is_available(at::TensorList tensors);

comm_init_rank(int nranks, const ncclUniqueId& comm_id, int rank);
 void comm_destroy(ncclComm_t comm);

void all_reduce(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = static_cast<int>(xcclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});
} // namespace torch::xpu::xccl

