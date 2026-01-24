#include <c10/core/DispatchKey.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/distributed/c10d/Functional.hpp>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/jit/frontend/schema_type_parser.h>
#include <torch/custom_class_detail.h>
#include <utility>

namespace {

const std::unordered_map<std::string, c10d::ReduceOp> str_to_reduce_op = {
    {"sum", c10d::ReduceOp(c10d::ReduceOp::RedOpType::SUM)},
    {"avg", c10d::ReduceOp(c10d::ReduceOp::RedOpType::AVG)},
    {"product", c10d::ReduceOp(c10d::ReduceOp::RedOpType::PRODUCT)},
    {"min", c10d::ReduceOp(c10d::ReduceOp::RedOpType::MIN)},
    {"max", c10d::ReduceOp(c10d::ReduceOp::RedOpType::MAX)},
    {"band", c10d::ReduceOp(c10d::ReduceOp::RedOpType::BAND)},
    {"bor", c10d::ReduceOp(c10d::ReduceOp::RedOpType::BOR)},
    {"bxor", c10d::ReduceOp(c10d::ReduceOp::RedOpType::BXOR)},
    // TODO: support premul_sum
    // {"premul_sum", c10d::ReduceOp(c10d::ReduceOp::RedOpType::PREMUL_SUM)},
    {"unused", c10d::ReduceOp(c10d::ReduceOp::RedOpType::UNUSED)}};

c10d::ReduceOp to_reduce_op(const std::string& reduce_op) {
  auto it = str_to_reduce_op.find(reduce_op);
  TORCH_CHECK(
      it != str_to_reduce_op.end(), "Unrecognized reduce_op: ", reduce_op);
  return it->second;
}

at::Tensor allocate_all_gather_output(
    const at::Tensor& input,
    int64_t group_size) {
  TORCH_CHECK(input.is_contiguous());
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
  TORCH_CHECK(input.is_contiguous());
  auto output_size = input.sizes().vec();
  if (output_size[0] % group_size != 0) {
    LOG(WARNING) << "The first dimension of the reduce_scatter input ("
                 << output_size[0] << ") is not divisible by the group size ("
                 << group_size << ").";
  }
  output_size[0] /= group_size;
  return at::empty(
      output_size,
      at::TensorOptions().dtype(input.dtype()).device(input.device()));
}

} // namespace

namespace c10d {

at::Tensor& all_reduce_(
    at::Tensor& input,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string reduce_op,
    c10::intrusive_ptr<c10d::ProcessGroup> group) {
  c10d::AllreduceOptions opts;
  opts.reduceOp = to_reduce_op(reduce_op);

  std::vector<at::Tensor> inputs{input};
  auto work = group->allreduce(inputs, opts);
  c10d::register_work(input, work);
  return input;
}

at::Tensor all_reduce(
    const at::Tensor& input,
    std::string reduce_op,
    c10::intrusive_ptr<c10d::ProcessGroup> group) {
  if (input.is_complex()) {
    TORCH_CHECK(
        // TODO - ideally use 'to_reduce_op' helper but it currently errors on
        // premul_sum
        reduce_op == "sum" || reduce_op == "avg" || reduce_op == "premul_sum" ||
            reduce_op == "unused",
        "all_reduce: reduce_op ",
        reduce_op,
        " does not support complex tensors");
  }
  auto input_real = input.is_complex() ? at::view_as_real(input) : input;
  auto output = input_real.clone(at::MemoryFormat::Contiguous);
  auto output_ret = all_reduce_(output, std::move(reduce_op), group);
  return input.is_complex() ? at::view_as_complex(output_ret) : output_ret;
}

at::Tensor& all_reduce_(
    at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
  auto group = c10d::resolve_process_group(group_name);
  return all_reduce_(input, std::move(reduce_op), group);
}

at::Tensor all_reduce(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
  auto group = c10d::resolve_process_group(group_name);
  return all_reduce(input, std::move(reduce_op), group);
}

std::vector<at::Tensor> all_reduce_coalesced_(
    std::vector<at::Tensor> inputs,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string reduce_op,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name) {
  auto group = c10d::resolve_process_group(group_name);
  return all_reduce_coalesced_(inputs, std::move(reduce_op), group);
}

std::vector<at::Tensor> all_reduce_coalesced_(
    std::vector<at::Tensor> inputs,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string reduce_op,
    c10::intrusive_ptr<c10d::ProcessGroup> group) {
  c10d::AllreduceCoalescedOptions opts;
  opts.reduceOp = to_reduce_op(reduce_op);

  auto work = group->allreduce_coalesced(inputs, opts);
  for (const auto& tensor : inputs) {
    c10d::register_work(tensor, work);
  }
  return inputs;
}

std::vector<at::Tensor> all_reduce_coalesced(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::vector<at::Tensor> inputs,
    std::string reduce_op,
    std::string group_name) {
  auto group = c10d::resolve_process_group(group_name);
  return all_reduce_coalesced(inputs, std::move(reduce_op), group);
}

std::vector<at::Tensor> all_reduce_coalesced(
    std::vector<at::Tensor> inputs,
    std::string reduce_op,
    c10::intrusive_ptr<c10d::ProcessGroup> group) {
  std::vector<at::Tensor> outputs;
  outputs.reserve(inputs.size());
  for (const auto& tensor : inputs) {
    outputs.push_back(tensor.clone(at::MemoryFormat::Contiguous));
  }
  return all_reduce_coalesced_(outputs, std::move(reduce_op), group);
}

std::vector<at::Tensor> all_gather_into_tensor_coalesced(
    std::vector<at::Tensor> inputs,
    int64_t group_size,
    c10::intrusive_ptr<c10d::ProcessGroup> group) {
  std::vector<at::Tensor> outputs;
  outputs.reserve(inputs.size());
  for (const auto& tensor : inputs) {
    TORCH_CHECK(tensor.is_contiguous());
    outputs.push_back(allocate_all_gather_output(tensor, group_size));
  }

  auto work = group->allgather_into_tensor_coalesced(outputs, inputs);
  for (const auto& tensor : outputs) {
    c10d::register_work(tensor, work);
  }
  return outputs;
}

at::Tensor all_gather_into_tensor(
    const at::Tensor& input,
    int64_t group_size,
    c10::intrusive_ptr<c10d::ProcessGroup> group) {
  TORCH_CHECK(input.is_contiguous());
  auto real_input = input.is_complex() ? at::view_as_real(input) : input;
  std::vector<at::Tensor> inputs{real_input};
  auto output = all_gather_into_tensor_coalesced(inputs, group_size, group)[0];
  return input.is_complex() ? at::view_as_complex(output) : output;
}

at::Tensor& all_gather_into_tensor_out(
    at::Tensor& input,
    int64_t group_size,
    c10::intrusive_ptr<c10d::ProcessGroup> group,
    at::Tensor& output) {
  TORCH_CHECK(input.is_contiguous());
  c10d::AllgatherOptions opts;

  auto work = group->_allgather_base(output, input, opts);
  c10d::register_work(output, work);
  return output;
}

std::vector<at::Tensor> all_gather_into_tensor_coalesced(
    std::vector<at::Tensor> inputs,
    int64_t group_size,
    std::string group_name) {
  auto group = c10d::resolve_process_group(group_name);
  return all_gather_into_tensor_coalesced(inputs, group_size, group);
}

at::Tensor all_gather_into_tensor(
    const at::Tensor& input,
    int64_t group_size,
    std::string group_name) {
  auto group = c10d::resolve_process_group(group_name);
  return all_gather_into_tensor(input, group_size, group);
}

at::Tensor& all_gather_into_tensor_out(
    at::Tensor& input,
    int64_t group_size,
    const std::string& group_name,
    at::Tensor& output) {
  auto group = c10d::resolve_process_group(group_name);
  return all_gather_into_tensor_out(input, group_size, group, output);
}

std::vector<at::Tensor> reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor> inputs,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string reduce_op,
    int64_t group_size,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name) {
  auto group = c10d::resolve_process_group(group_name);
  return reduce_scatter_tensor_coalesced(
      inputs, std::move(reduce_op), group_size, group);
}

std::vector<at::Tensor> reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor> inputs,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string reduce_op,
    int64_t group_size,
    c10::intrusive_ptr<c10d::ProcessGroup> group) {
  c10d::ReduceScatterOptions opts;
  opts.reduceOp = to_reduce_op(reduce_op);
  std::vector<at::Tensor> outputs;
  outputs.reserve(inputs.size());
  for (const auto& tensor : inputs) {
    TORCH_CHECK(tensor.is_contiguous());
    outputs.push_back(allocate_reduce_scatter_output(tensor, group_size));
  }

  auto work = group->reduce_scatter_tensor_coalesced(outputs, inputs, opts);
  for (const auto& tensor : outputs) {
    c10d::register_work(tensor, work);
  }
  return outputs;
}

static std::vector<at::Tensor> reduce_scatter_tensor_coalesced_out(
    std::vector<at::Tensor> inputs,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string reduce_op,
    int64_t group_size,
    c10::intrusive_ptr<c10d::ProcessGroup> group,
    std::vector<at::Tensor>& outputs) {
  c10d::ReduceScatterOptions opts;
  opts.reduceOp = to_reduce_op(reduce_op);

  auto work = group->reduce_scatter_tensor_coalesced(outputs, inputs, opts);
  for (const auto& tensor : outputs) {
    c10d::register_work(tensor, work);
  }
  return outputs;
}

// static std::vector<at::Tensor> reduce_scatter_tensor_coalesced_out(
//     std::vector<at::Tensor> inputs,
//     // NOLINTNEXTLINE(performance-unnecessary-value-param)
//     std::string reduce_op,
//     int64_t group_size,
//     // NOLINTNEXTLINE(performance-unnecessary-value-param)
//     std::string group_name,
//     std::vector<at::Tensor>& outputs) {
//   auto group = c10d::resolve_process_group(std::move(group_name));
//   return reduce_scatter_tensor_coalesced_out(
//       inputs, std::move(reduce_op), group_size, group, outputs);
// }

at::Tensor reduce_scatter_tensor(
    const at::Tensor& input,
    std::string reduce_op,
    int64_t group_size,
    std::string group_name) {
  auto group = c10d::resolve_process_group(group_name);
  return reduce_scatter_tensor(input, std::move(reduce_op), group_size, group);
}

at::Tensor reduce_scatter_tensor(
    const at::Tensor& input,
    std::string reduce_op,
    int64_t group_size,
    c10::intrusive_ptr<c10d::ProcessGroup> group) {
  TORCH_CHECK(input.is_contiguous());
  if (input.is_complex()) {
    auto real_input = at::view_as_real(input);
    std::vector<at::Tensor> inputs{real_input};
    return at::view_as_complex(reduce_scatter_tensor_coalesced(
        inputs, std::move(reduce_op), group_size, group)[0]);
  }
  std::vector<at::Tensor> inputs{input};
  return reduce_scatter_tensor_coalesced(
      inputs, std::move(reduce_op), group_size, group)[0];
}

at::Tensor reduce_scatter_tensor_out(
    const at::Tensor& input,
    std::string reduce_op,
    int64_t group_size,
    std::string group_name,
    at::Tensor& output) {
  auto group = c10d::resolve_process_group(group_name);
  return reduce_scatter_tensor_out(
      input, std::move(reduce_op), group_size, group, output);
}

at::Tensor reduce_scatter_tensor_out(
    const at::Tensor& input,
    std::string reduce_op,
    int64_t group_size,
    c10::intrusive_ptr<c10d::ProcessGroup> group,
    at::Tensor& output) {
  TORCH_CHECK(input.is_contiguous());
  if (input.is_complex()) {
    TORCH_CHECK(output.is_complex())
    auto real_input = at::view_as_real(input);
    std::vector<at::Tensor> inputs{std::move(real_input)};
    auto real_output = at::view_as_real(output);
    std::vector<at::Tensor> outputs{std::move(real_output)};
    return at::view_as_complex(reduce_scatter_tensor_coalesced_out(
        inputs, std::move(reduce_op), group_size, group, outputs)[0]);
  }
  std::vector<at::Tensor> inputs{std::move(input)};
  std::vector<at::Tensor> outputs{std::move(output)};
  return reduce_scatter_tensor_coalesced_out(
      inputs, std::move(reduce_op), group_size, group, outputs)[0];
}

at::Tensor all_to_all_single(
    const at::Tensor& input,
    c10::SymIntArrayRef _output_split_sizes,
    c10::SymIntArrayRef _input_split_sizes,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name) {
  auto group = c10d::resolve_process_group(std::move(group_name));
  return all_to_all_single(
      input, _output_split_sizes, _input_split_sizes, std::move(group));
}

at::Tensor all_to_all_single(
    const at::Tensor& input,
    c10::SymIntArrayRef _output_split_sizes,
    c10::SymIntArrayRef _input_split_sizes,
    c10::intrusive_ptr<ProcessGroup> group) {
  std::vector<int64_t> output_split_sizes;
  std::vector<int64_t> input_split_sizes;
  output_split_sizes.reserve(_output_split_sizes.size());
  input_split_sizes.reserve(_input_split_sizes.size());
  for (const auto& size : _output_split_sizes) {
    output_split_sizes.emplace_back(size.expect_int());
  }
  for (const auto& size : _input_split_sizes) {
    input_split_sizes.emplace_back(size.expect_int());
  }

  TORCH_CHECK(input.is_contiguous());
  std::vector<int64_t> output_sizes = input.sizes().vec();
  output_sizes[0] = std::accumulate(
      output_split_sizes.begin(), output_split_sizes.end(), int64_t(0));
  auto output = input.new_empty(output_sizes);

  auto work = group->alltoall_base(
      output,
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<at::Tensor&>(input),
      output_split_sizes,
      input_split_sizes);
  c10d::register_work(output, work);
  return output;
}

namespace {

at::Tensor all_to_all_single_dispatch(
    const at::Tensor& input,
    c10::SymIntArrayRef output_split_sizes,
    c10::SymIntArrayRef input_split_sizes,
    const c10::IValue& group) {
  c10::intrusive_ptr<c10d::ProcessGroup> pg;
  if (group.isString()) {
    return c10d::all_to_all_single(
        input, output_split_sizes, input_split_sizes, group.toStringRef());
  } else if (group.isCapsule()) {
    pg = c10::static_intrusive_pointer_cast<c10d::ProcessGroup>(
        group.toCapsule());
    return c10d::all_to_all_single(
        input, output_split_sizes, input_split_sizes, pg);
  } else {
    TORCH_CHECK(
        false,
        "all_to_all_single(): argument 'group' must be either a string (group name) "
        "or a ProcessGroup object, but got ",
        group.type()->str());
  }
}

} // anonymous namespace

// NOLINTNEXTLINE(performance-unnecessary-value-param)
at::Tensor& broadcast_(at::Tensor& input, int64_t src, std::string group_name) {
  auto group = c10d::resolve_process_group(group_name);
  return broadcast_(input, src, group);
}

at::Tensor& broadcast_(
    at::Tensor& input,
    int64_t src,
    c10::intrusive_ptr<c10d::ProcessGroup> group) {
  c10d::BroadcastOptions opts;
  opts.rootRank = src;
  auto input_real = input.is_complex() ? at::view_as_real(input) : input;
  std::vector<at::Tensor> inputs{input_real};

  auto work = group->broadcast(inputs, opts);
  c10d::register_work(input, work);
  return input;
}

at::Tensor broadcast(
    const at::Tensor& input,
    int64_t src,
    std::string group_name) {
  auto group = c10d::resolve_process_group(group_name);
  return broadcast(input, src, group);
}

at::Tensor broadcast(
    const at::Tensor& input,
    int64_t src,
    c10::intrusive_ptr<c10d::ProcessGroup> group) {
  auto output = input.clone(at::MemoryFormat::Contiguous);
  return broadcast_(output, src, group);
}

} // namespace c10d

namespace {

c10::intrusive_ptr<c10d::ProcessGroup> get_process_group(
    const c10::IValue& group,
    const char* func_name) {
  if (group.isString()) {
    return c10d::resolve_process_group(group.toStringRef());
  } else if (group.isCapsule()) {
    return c10::static_intrusive_pointer_cast<c10d::ProcessGroup>(
        group.toCapsule());
  } else {
    TORCH_CHECK(
        false,
        func_name,
        "(): argument 'group' must be either a string (group name) "
        "or a ProcessGroup object, but got ",
        group.type()->str());
  }
}

at::Tensor& all_reduce__dispatch(
    at::Tensor& input,
    std::string reduce_op,
    const c10::IValue& group) {
  return c10d::all_reduce_(
      input, std::move(reduce_op), get_process_group(group, "all_reduce_"));
}

at::Tensor all_reduce_dispatch(
    const at::Tensor& input,
    std::string reduce_op,
    const c10::IValue& group) {
  return c10d::all_reduce(
      input, std::move(reduce_op), get_process_group(group, "all_reduce"));
}

std::vector<at::Tensor> all_gather_into_tensor_coalesced_dispatch(
    std::vector<at::Tensor> inputs,
    int64_t group_size,
    const c10::IValue& group) {
  return c10d::all_gather_into_tensor_coalesced(
      inputs,
      group_size,
      get_process_group(group, "all_gather_into_tensor_coalesced"));
}

at::Tensor all_gather_into_tensor_dispatch(
    const at::Tensor& input,
    int64_t group_size,
    const c10::IValue& group) {
  return c10d::all_gather_into_tensor(
      input, group_size, get_process_group(group, "all_gather_into_tensor"));
}

at::Tensor& all_gather_into_tensor_out_dispatch(
    at::Tensor& input,
    int64_t group_size,
    const c10::IValue& group,
    at::Tensor& output) {
  return c10d::all_gather_into_tensor_out(
      input,
      group_size,
      get_process_group(group, "all_gather_into_tensor_out"),
      output);
}

std::vector<at::Tensor> all_reduce_coalesced__dispatch(
    std::vector<at::Tensor> inputs,
    std::string reduce_op,
    const c10::IValue& group) {
  return c10d::all_reduce_coalesced_(
      inputs,
      std::move(reduce_op),
      get_process_group(group, "all_reduce_coalesced_"));
}

std::vector<at::Tensor> all_reduce_coalesced_dispatch(
    std::vector<at::Tensor> inputs,
    std::string reduce_op,
    const c10::IValue& group) {
  return c10d::all_reduce_coalesced(
      inputs,
      std::move(reduce_op),
      get_process_group(group, "all_reduce_coalesced"));
}

std::vector<at::Tensor> reduce_scatter_tensor_coalesced_dispatch(
    std::vector<at::Tensor> inputs,
    std::string reduce_op,
    int64_t group_size,
    const c10::IValue& group) {
  return c10d::reduce_scatter_tensor_coalesced(
      inputs,
      std::move(reduce_op),
      group_size,
      get_process_group(group, "reduce_scatter_tensor_coalesced"));
}

at::Tensor reduce_scatter_tensor_dispatch(
    const at::Tensor& input,
    std::string reduce_op,
    int64_t group_size,
    const c10::IValue& group) {
  return c10d::reduce_scatter_tensor(
      input,
      std::move(reduce_op),
      group_size,
      get_process_group(group, "reduce_scatter_tensor"));
}

at::Tensor reduce_scatter_tensor_out_dispatch(
    const at::Tensor& input,
    std::string reduce_op,
    int64_t group_size,
    const c10::IValue& group,
    at::Tensor& output) {
  return c10d::reduce_scatter_tensor_out(
      input,
      std::move(reduce_op),
      group_size,
      get_process_group(group, "reduce_scatter_tensor_out"),
      output);
}

at::Tensor& broadcast__dispatch(
    at::Tensor& input,
    int64_t src,
    const c10::IValue& group) {
  return c10d::broadcast_(input, src, get_process_group(group, "broadcast_"));
}

at::Tensor broadcast_dispatch(
    const at::Tensor& input,
    int64_t src,
    const c10::IValue& group) {
  return c10d::broadcast(input, src, get_process_group(group, "broadcast"));
}

} // namespace

TORCH_LIBRARY(_c10d_functional, m) {
  m.def(
      "all_reduce(Tensor input, str reduce_op, Any group) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, all_reduce_dispatch),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "all_reduce_(Tensor(a!) input, str reduce_op, Any group) -> Tensor(a!)",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, all_reduce__dispatch),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "all_reduce_coalesced(Tensor[] inputs, str reduce_op, Any group) -> Tensor[]",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          all_reduce_coalesced_dispatch),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "all_reduce_coalesced_(Tensor[](a!) inputs, str reduce_op, Any group) -> Tensor[](a!)",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          all_reduce_coalesced__dispatch),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "all_gather_into_tensor_out(Tensor input, int group_size, Any group, *, Tensor(a!) out) -> Tensor(a!)",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          all_gather_into_tensor_out_dispatch),
      {at::Tag::pt2_compliant_tag, at::Tag::needs_contiguous_strides});

  m.def(
      "all_gather_into_tensor(Tensor input, int group_size, Any group) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          all_gather_into_tensor_dispatch),
      {at::Tag::pt2_compliant_tag, at::Tag::needs_contiguous_strides});

  m.def(
      "all_gather_into_tensor_coalesced(Tensor[] inputs, int group_size, Any group) -> Tensor[]",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          all_gather_into_tensor_coalesced_dispatch),
      {at::Tag::pt2_compliant_tag, at::Tag::needs_contiguous_strides});

  m.def(
      "reduce_scatter_tensor(Tensor input, str reduce_op, int group_size, Any group) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          reduce_scatter_tensor_dispatch),
      {at::Tag::pt2_compliant_tag, at::Tag::needs_contiguous_strides});

  m.def(
      "reduce_scatter_tensor_out(Tensor input, str reduce_op, int group_size, Any group, *, Tensor(a!) out) -> Tensor(a!)",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          reduce_scatter_tensor_out_dispatch),
      {at::Tag::pt2_compliant_tag, at::Tag::needs_contiguous_strides});

  m.def(
      "reduce_scatter_tensor_coalesced(Tensor[] inputs, str reduce_op, int group_size, Any group) -> Tensor[]",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          reduce_scatter_tensor_coalesced_dispatch),
      {at::Tag::pt2_compliant_tag, at::Tag::needs_contiguous_strides});

  m.def(
      "all_to_all_single("
      "Tensor input, "
      "SymInt[] output_split_sizes, "
      "SymInt[] input_split_sizes, "
      "Any group) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          c10d::all_to_all_single_dispatch),
      {at::Tag::pt2_compliant_tag, at::Tag::needs_contiguous_strides});

  m.def(
      "broadcast(Tensor input, int src, Any group) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, broadcast_dispatch),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "broadcast_(Tensor(a!) input, int src, Any group) -> Tensor(a!)",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, broadcast__dispatch),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "wait_tensor(Tensor tensor) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, c10d::wait_tensor),
      {at::Tag::pt2_compliant_tag});
}

namespace {
class AllToAllSingle : public torch::autograd::Function<AllToAllSingle> {
 public:
  static torch::autograd::Variable forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      // NOLINTNEXTLINE(performance-unnecessary-value-param)
      at::SymIntArrayRef output_split_sizes,
      // NOLINTNEXTLINE(performance-unnecessary-value-param)
      at::SymIntArrayRef input_split_sizes,
      const c10::IValue& group) {
    // swap sizes for backwards pass
    ctx->saved_data["output_split_sizes"] = input_split_sizes.vec();
    ctx->saved_data["input_split_sizes"] = output_split_sizes.vec();
    ctx->saved_data["group"] = group;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("_c10d_functional::all_to_all_single", "")
        .typed<decltype(c10d::all_to_all_single_dispatch)>()
        .call(input, output_split_sizes, input_split_sizes, group);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_out_list) {
    std::vector<c10::SymInt> output_split_sizes =
        ctx->saved_data["output_split_sizes"].toSymIntVector();
    std::vector<c10::SymInt> input_split_sizes =
        ctx->saved_data["input_split_sizes"].toSymIntVector();
    auto group = ctx->saved_data["group"];

    DCHECK(grad_out_list.size() == 1);
    auto grad_out = grad_out_list[0].contiguous();

    auto out =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("_c10d_functional::all_to_all_single", "")
            .typed<decltype(c10d::all_to_all_single_dispatch)>()
            .call(grad_out, output_split_sizes, input_split_sizes, group);

    // do an explicit wait to avoid cuda stream issues
    // TODO: track active cuda stream in wait
    out = c10::Dispatcher::singleton()
              .findSchemaOrThrow("_c10d_functional::wait_tensor", "")
              .typed<decltype(c10d::wait_tensor)>()
              .call(out);

    return {out, at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

at::Tensor all_to_all_single_autograd(
    const at::Tensor& input,
    at::SymIntArrayRef output_split_sizes,
    at::SymIntArrayRef input_split_sizes,
    const c10::IValue& group) {
  return AllToAllSingle::apply(
      input, output_split_sizes, input_split_sizes, group);
}

namespace {

at::Tensor all_to_all_single_autograd_dispatch(
    const at::Tensor& input,
    c10::SymIntArrayRef output_split_sizes,
    c10::SymIntArrayRef input_split_sizes,
    const c10::IValue& group) {
  return ::all_to_all_single_autograd(
      input, output_split_sizes, input_split_sizes, group);
}

} // anonymous namespace

class ReduceScatterTensor
    : public torch::autograd::Function<ReduceScatterTensor> {
 public:
  static torch::autograd::Variable forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const std::string& reduce_op,
      int64_t group_size,
      const std::string& group_name) {
    TORCH_CHECK(reduce_op == "sum", "Only sum reduce op is supported");

    ctx->saved_data["group_size"] = group_size;
    ctx->saved_data["group_name"] = group_name;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("_c10d_functional::reduce_scatter_tensor", "")
        .typed<at::Tensor(
            const at::Tensor&, std::string, int64_t, const c10::IValue&)>()
        .call(input, reduce_op, group_size, group_name);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_out_list) {
    const int64_t group_size = ctx->saved_data["group_size"].toInt();
    const std::string& group_name = ctx->saved_data["group_name"].toStringRef();

    DCHECK(grad_out_list.size() == 1);
    const auto& grad_out = grad_out_list[0];

    auto out =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("_c10d_functional::all_gather_into_tensor", "")
            .typed<at::Tensor(const at::Tensor&, int64_t, const c10::IValue&)>()
            .call(grad_out, group_size, group_name);

    // do an explicit wait to avoid cuda stream issues
    // TODO: track active cuda stream in wait
    out = c10::Dispatcher::singleton()
              .findSchemaOrThrow("_c10d_functional::wait_tensor", "")
              .typed<decltype(c10d::wait_tensor)>()
              .call(out);

    return {
        out,
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
    };
  }
};

at::Tensor reduce_scatter_tensor_autograd(
    const at::Tensor& input,
    const std::string& reduce_op,
    int64_t group_size,
    const std::string& group_name) {
  return ReduceScatterTensor::apply(input, reduce_op, group_size, group_name);
}

class AllGatherIntoTensor
    : public torch::autograd::Function<AllGatherIntoTensor> {
 public:
  static torch::autograd::Variable forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      int64_t group_size,
      const std::string& group_name) {
    ctx->saved_data["group_size"] = group_size;
    ctx->saved_data["group_name"] = group_name;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("_c10d_functional::all_gather_into_tensor", "")
        .typed<at::Tensor(const at::Tensor&, int64_t, const c10::IValue&)>()
        .call(input, group_size, group_name);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_out_list) {
    const int64_t group_size = ctx->saved_data["group_size"].toInt();
    const std::string& group_name = ctx->saved_data["group_name"].toStringRef();

    DCHECK(grad_out_list.size() == 1);
    const auto& grad_out = grad_out_list[0];

    auto out =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("_c10d_functional::reduce_scatter_tensor", "")
            .typed<at::Tensor(
                const at::Tensor&, std::string, int64_t, const c10::IValue&)>()
            .call(grad_out, "sum", group_size, group_name);

    // do an explicit wait to avoid cuda stream issues
    // TODO: track active cuda stream in wait
    out = c10::Dispatcher::singleton()
              .findSchemaOrThrow("_c10d_functional::wait_tensor", "")
              .typed<decltype(c10d::wait_tensor)>()
              .call(out);

    return {
        out,
        at::Tensor(),
        at::Tensor(),
    };
  }
};

at::Tensor all_gather_into_tensor_autograd(
    const at::Tensor& input,
    int64_t group_size,
    const std::string& group_name) {
  return AllGatherIntoTensor::apply(input, group_size, group_name);
}

} // namespace

TORCH_LIBRARY(_c10d_functional_autograd, m) {
  m.def(
      "all_to_all_single("
      "Tensor input, "
      "SymInt[] output_split_sizes, "
      "SymInt[] input_split_sizes, "
      "Any group) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::Autograd, ::all_to_all_single_autograd_dispatch),
      {at::Tag::pt2_compliant_tag});
  m.def(
      "reduce_scatter_tensor("
      "Tensor input, "
      "str reduce_op, "
      "int group_size, "
      "str group_name) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::Autograd, ::reduce_scatter_tensor_autograd),
      {at::Tag::pt2_compliant_tag});
  m.def(
      "all_gather_into_tensor("
      "Tensor input, "
      "int group_size, "
      "str group_name) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::Autograd, ::all_gather_into_tensor_autograd),
      {at::Tag::pt2_compliant_tag});
}

namespace {
// DTensor related comm operations, sharing code with functional collective for
// now
at::Tensor shard_dim_alltoall(
    const at::Tensor& input,
    int64_t gather_dim,
    int64_t shard_dim,
    c10::intrusive_ptr<c10d::ProcessGroup> group) {
  auto group_size = group->getSize();
  std::vector<int64_t> input_sizes = input.sizes().vec();
  std::vector<int64_t> output_sizes = input.sizes().vec();
  if (input_sizes[shard_dim] % group_size != 0) {
    LOG(WARNING) << "The shard dimension of the shard_dim_alltoall input ("
                 << input_sizes[shard_dim]
                 << ") is not divisible by the group size (" << group_size
                 << ").";
  }
  input_sizes[shard_dim] /= group_size;
  input_sizes.insert(input_sizes.begin() + shard_dim, group_size);

  auto tensor_reshaped = input.view(input_sizes);
  auto tensor_shard_contig = tensor_reshaped.movedim(shard_dim, 0).contiguous();
  auto tensor_for_comm = input.is_complex()
      ? at::view_as_real(tensor_shard_contig)
      : tensor_shard_contig;

  auto recv_tensor = at::empty_like(tensor_for_comm);
  std::vector<int64_t> out_split_sizes;
  std::vector<int64_t> in_split_sizes;
  c10d::AllToAllOptions opts;

  auto work = group->alltoall_base(
      recv_tensor, tensor_for_comm, out_split_sizes, in_split_sizes, opts);

  // TODO: it's tricky to get the current async behavior work for shard dim
  // alltoall so for now we just keep this comm op to be synchronous. We might
  // need to have sth similar to future callback to do the permute, contiguous
  // and view calls. We can revisit later how to support the async case with the
  // Work registry.
  work->wait();

  auto output = recv_tensor.movedim(0, gather_dim).contiguous();

  // view/reshape it back to the expected output shape
  output_sizes[shard_dim] /= group_size;
  output_sizes[gather_dim] *= group_size;
  return input.is_complex() ? at::view_as_complex(output).view(output_sizes)
                            : output.view(output_sizes);
}

at::Tensor shard_dim_alltoall(
    const at::Tensor& input,
    int64_t gather_dim,
    int64_t shard_dim,
    const std::string& group_name) {
  auto group = c10d::resolve_process_group(group_name);
  return shard_dim_alltoall(input, gather_dim, shard_dim, std::move(group));
}

at::Tensor shard_dim_alltoall_dispatch(
    const at::Tensor& input,
    int64_t gather_dim,
    int64_t shard_dim,
    const c10::IValue& group) {
  c10::intrusive_ptr<c10d::ProcessGroup> pg;
  if (group.isString()) {
    return shard_dim_alltoall(
        input, gather_dim, shard_dim, group.toStringRef());
  } else if (group.isCapsule()) {
    pg = c10::static_intrusive_pointer_cast<c10d::ProcessGroup>(
        group.toCapsule());
    return shard_dim_alltoall(input, gather_dim, shard_dim, pg);
  } else {
    TORCH_CHECK(
        false,
        "shard_dim_alltoall(): argument 'group' must be either a string (group name) "
        "or a ProcessGroup object, but got ",
        group.type()->str());
  }
}

} // namespace

// DTensor comm op registry
TORCH_LIBRARY(_dtensor, m) {
  m.def(
      "shard_dim_alltoall(Tensor input, int gather_dim, int shard_dim, Any group) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          shard_dim_alltoall_dispatch),
      {at::Tag::pt2_compliant_tag});
}
