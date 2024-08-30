#pragma once

#include <ATen/ATen.h>
#include <oneapi/ccl.hpp>
#include <cstddef>
#include <optional>
#include <vector>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace torch::xpu::xccl {

using xcclComm_t = ccl::communicator;

using XCCL_KVS = ccl::shared_ptr_class<ccl::kvs>;

extern XCCL_KVS kvs;

XCCL_KVS get_kvs(int rank, c10d::Store& store);

enum class xcclRedOp { Sum = 0, Prod = 1, Max = 2, Min = 3 };

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

at::ArrayRef<xcclComm_t> get_communicators(at::TensorList inputs);
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

// using comm_list = std::vector<xor>;
// using stream_list = std::vector<std::optional<at::xpu::XPUStream>>;

std::uint64_t version();
const char* version_suffix();

bool is_available(at::TensorList tensors);

// comm_init_rank(int nranks, const ncclUniqueId& comm_id, int rank);
// void comm_destroy(xcclComm_t comm);

// void all_reduce(
//     const std::vector<at::Tensor>& inputs,
//     std::vector<at::Tensor>& outputs,
//     int32_t op = static_cast<int>(xcclRedOp::Sum),
//     const stream_list& streams = {},
//     const comm_list& user_comms = {});
} // namespace torch::xpu::xccl
