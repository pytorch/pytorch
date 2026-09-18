#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/library.h>

#include <algorithm>
#include <numeric>

namespace {

const std::unordered_map<std::string, c10d::ReduceOp::RedOpType>
    str_to_reduce_op = {
        {"sum", c10d::ReduceOp::RedOpType::SUM},
        {"avg", c10d::ReduceOp::RedOpType::AVG},
        {"product", c10d::ReduceOp::RedOpType::PRODUCT},
        {"min", c10d::ReduceOp::RedOpType::MIN},
        {"max", c10d::ReduceOp::RedOpType::MAX},
        {"band", c10d::ReduceOp::RedOpType::BAND},
        {"bor", c10d::ReduceOp::RedOpType::BOR},
        {"bxor", c10d::ReduceOp::RedOpType::BXOR},
        {"premul_sum", c10d::ReduceOp::RedOpType::PREMUL_SUM},
        {"unused", c10d::ReduceOp::RedOpType::UNUSED}};

c10::intrusive_ptr<c10d::ReduceOp> to_reduce_op(const std::string& reduce_op) {
  auto it = str_to_reduce_op.find(reduce_op);
  TORCH_CHECK(
      it != str_to_reduce_op.end(), "Unrecognized reduce_op: ", reduce_op);
  return c10::make_intrusive<c10d::ReduceOp>(it->second);
}

at::Tensor allocate_all_gather_output(
    const at::Tensor& input,
    int64_t group_size) {
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  auto output_size = input.sizes().vec();
  if (output_size.empty()) {
    output_size.push_back(group_size);
  } else {
    output_size[0] *= group_size;
  }
  return at::empty(
      output_size,
      at::TensorOptions().dtype(input.dtype()).device(input.device()));
}

at::Tensor allocate_reduce_scatter_output(
    const at::Tensor& input,
    const int64_t group_size) {
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  auto output_size = input.sizes().vec();
  TORCH_CHECK(
      !output_size.empty() && output_size[0] % group_size == 0,
      "reduce_scatter input must have a first dimension divisible by group_size");
  output_size[0] /= group_size;
  return at::empty(
      output_size,
      at::TensorOptions().dtype(input.dtype()).device(input.device()));
}

c10::intrusive_ptr<c10d::ProcessGroup> get_process_group(
    const c10::IValue& group_name,
    const char* func_name) {
  if (group_name.isString()) {
    return c10d::resolve_process_group(group_name.toStringRef());
  }
  if (group_name.isCapsule()) {
    return c10::static_intrusive_pointer_cast<c10d::ProcessGroup>(
        group_name.toCapsule());
  }
  TORCH_CHECK(
      false,
      func_name,
      "(): argument 'group_name' must be either a string (group name) "
      "or a ProcessGroup object, but got ",
      group_name.type()->str());
}

c10::intrusive_ptr<c10d::ReduceOp> get_reduce_op(
    const c10::IValue& op_name,
    const char* func_name) {
  if (op_name.isString()) {
    return to_reduce_op(op_name.toStringRef());
  }
  if (op_name.isCapsule()) {
    return c10::static_intrusive_pointer_cast<c10d::ReduceOp>(
        op_name.toCapsule());
  }
  TORCH_CHECK(
      false,
      func_name,
      "(): argument 'reduce_op' must be either a string (op name) "
      "or a ReduceOp object, but got ",
      op_name.type()->str());
}

// Configured collectives use opaque compiler fallback. Complete their stream
// dependency before returning: fallback does not model asynchronous input
// reads.
at::Tensor all_reduce_config(
    const at::Tensor& input,
    const c10::IValue& reduce_op,
    const c10::IValue& group_name,
    const c10::Dict<std::string, c10::IValue>& config) {
  auto op = get_reduce_op(reduce_op, "all_reduce_config");
  TORCH_CHECK(
      !input.is_complex() || *op == c10d::ReduceOp::SUM ||
          *op == c10d::ReduceOp::AVG || *op == c10d::ReduceOp::PREMUL_SUM,
      "all_reduce: reduce_op does not support complex tensors");
  auto real_input = input.is_complex() ? at::view_as_real(input) : input;
  auto output = real_input.clone(at::MemoryFormat::Contiguous);
  std::vector<at::Tensor> tensors{output};
  c10d::AllreduceOptions opts;
  opts.reduceOp = *op;
  opts.config = config;
  get_process_group(group_name, "all_reduce_config")
      ->allreduce(tensors, opts)
      ->wait();
  return input.is_complex() ? at::view_as_complex(output) : output;
}

at::Tensor all_gather_config(
    const at::Tensor& input,
    int64_t group_size,
    const c10::IValue& group_name,
    const c10::Dict<std::string, c10::IValue>& config) {
  auto real_input = input.is_complex() ? at::view_as_real(input) : input;
  auto contiguous = real_input.contiguous();
  auto output = allocate_all_gather_output(contiguous, group_size);
  c10d::AllgatherOptions opts;
  opts.config = config;
  get_process_group(group_name, "all_gather_config")
      ->all_gather_single(output, contiguous, opts)
      ->wait();
  return input.is_complex() ? at::view_as_complex(output) : output;
}

at::Tensor reduce_scatter_config(
    const at::Tensor& input,
    const c10::IValue& reduce_op,
    int64_t group_size,
    const c10::IValue& group_name,
    const c10::Dict<std::string, c10::IValue>& config) {
  auto op = get_reduce_op(reduce_op, "reduce_scatter_config");
  TORCH_CHECK(
      !input.is_complex() || *op == c10d::ReduceOp::SUM ||
          *op == c10d::ReduceOp::AVG || *op == c10d::ReduceOp::PREMUL_SUM,
      "reduce_scatter: reduce_op does not support complex tensors");
  auto real_input = input.is_complex() ? at::view_as_real(input) : input;
  auto contiguous = real_input.contiguous();
  auto output = allocate_reduce_scatter_output(contiguous, group_size);
  c10d::ReduceScatterOptions opts;
  opts.reduceOp = *op;
  opts.config = config;
  get_process_group(group_name, "reduce_scatter_config")
      ->reduce_scatter_single(output, contiguous, opts)
      ->wait();
  return input.is_complex() ? at::view_as_complex(output) : output;
}

at::Tensor all_to_all_config(
    const at::Tensor& input,
    c10::SymIntArrayRef output_splits,
    c10::SymIntArrayRef input_splits,
    const c10::IValue& group_name,
    const c10::Dict<std::string, c10::IValue>& config) {
  std::vector<int64_t> outputs;
  std::vector<int64_t> inputs;
  for (const auto& size : output_splits) {
    outputs.push_back(size.expect_int());
  }
  for (const auto& size : input_splits) {
    inputs.push_back(size.expect_int());
  }
  auto contiguous = input.contiguous();
  auto shape = input.sizes().vec();
  TORCH_CHECK(!shape.empty(), "all_to_all_single requires a non-scalar tensor");
  shape[0] = std::accumulate(outputs.begin(), outputs.end(), int64_t(0));
  auto output = input.new_empty(shape);
  c10d::AllToAllOptions opts;
  opts.config = config;
  auto group = get_process_group(group_name, "all_to_all_config");
  if (inputs == outputs &&
      inputs.size() == static_cast<size_t>(group->getSize()) &&
      shape[0] == input.size(0) && !inputs.empty() &&
      std::all_of(inputs.begin(), inputs.end(), [&](int64_t size) {
        return size == inputs.front();
      })) {
    inputs.clear();
    outputs.clear();
  }
  group->all_to_all_single(output, contiguous, outputs, inputs, opts)->wait();
  return output;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(_c10d_functional, m) {
  m.def(
      "all_reduce_config(Tensor input, Any reduce_op, Any group_name, Dict(str, Any) config) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, all_reduce_config),
      {at::Tag::pt2_compliant_tag});
  m.def(
      "all_gather_into_tensor_config(Tensor input, int group_size, Any group_name, Dict(str, Any) config) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, all_gather_config),
      {at::Tag::pt2_compliant_tag});
  m.def(
      "reduce_scatter_tensor_config(Tensor input, Any reduce_op, int group_size, Any group_name, Dict(str, Any) config) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, reduce_scatter_config),
      {at::Tag::pt2_compliant_tag});
  m.def(
      "all_to_all_single_config(Tensor input, SymInt[] output_split_sizes, SymInt[] input_split_sizes, Any group_name, Dict(str, Any) config) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, all_to_all_config),
      {at::Tag::pt2_compliant_tag});
}
