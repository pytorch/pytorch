#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/DispatchKey.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/distributed/c10d/Functional.hpp>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
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
  if (output_size.size() == 0) {
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
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name) {
  c10d::AllreduceOptions opts;
  opts.reduceOp = to_reduce_op(reduce_op);

  std::vector<at::Tensor> inputs{input};
  auto group = c10d::resolve_process_group(group_name);
  auto work = group->allreduce(inputs, opts);
  c10d::register_work(input, work);
  return input;
}

at::Tensor all_reduce(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
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
  auto output_ret =
      all_reduce_(output, std::move(reduce_op), std::move(group_name));
  return input.is_complex() ? at::view_as_complex(output_ret) : output_ret;
}

std::vector<at::Tensor> all_reduce_coalesced_(
    std::vector<at::Tensor> inputs,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string reduce_op,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name) {
  c10d::AllreduceCoalescedOptions opts;
  opts.reduceOp = to_reduce_op(reduce_op);

  auto group = c10d::resolve_process_group(group_name);
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
  std::vector<at::Tensor> outputs;
  outputs.reserve(inputs.size());
  for (const auto& tensor : inputs) {
    outputs.push_back(tensor.clone(at::MemoryFormat::Contiguous));
  }
  return all_reduce_coalesced_(
      outputs, std::move(reduce_op), std::move(group_name));
}

std::vector<at::Tensor> all_gather_into_tensor_coalesced(
    std::vector<at::Tensor> inputs,
    int64_t group_size,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name) {
  std::vector<at::Tensor> outputs;
  outputs.reserve(inputs.size());
  for (const auto& tensor : inputs) {
    TORCH_CHECK(tensor.is_contiguous());
    outputs.push_back(allocate_all_gather_output(tensor, group_size));
  }

  auto group = c10d::resolve_process_group(group_name);
  auto work = group->allgather_into_tensor_coalesced(outputs, inputs);
  for (const auto& tensor : outputs) {
    c10d::register_work(tensor, work);
  }
  return outputs;
}

at::Tensor all_gather_into_tensor(
    const at::Tensor& input,
    int64_t group_size,
    std::string group_name) {
  TORCH_CHECK(input.is_contiguous());
  auto real_input = input.is_complex() ? at::view_as_real(input) : input;
  std::vector<at::Tensor> inputs{real_input};
  auto output = all_gather_into_tensor_coalesced(
      inputs, group_size, std::move(group_name))[0];
  return input.is_complex() ? at::view_as_complex(output) : output;
}

at::Tensor& all_gather_into_tensor_out(
    at::Tensor& input,
    int64_t group_size,
    const std::string& group_name,
    at::Tensor& output) {
  TORCH_CHECK(input.is_contiguous());
  c10d::AllgatherOptions opts;

  auto group = c10d::resolve_process_group(group_name);
  auto work = group->_allgather_base(output, input, opts);
  c10d::register_work(output, work);
  return output;
}

std::vector<at::Tensor> reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor> inputs,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string reduce_op,
    int64_t group_size,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name) {
  c10d::ReduceScatterOptions opts;
  opts.reduceOp = to_reduce_op(reduce_op);
  std::vector<at::Tensor> outputs;
  outputs.reserve(inputs.size());
  for (const auto& tensor : inputs) {
    TORCH_CHECK(tensor.is_contiguous());
    outputs.push_back(allocate_reduce_scatter_output(tensor, group_size));
  }

  auto group = c10d::resolve_process_group(group_name);
  auto work = group->reduce_scatter_tensor_coalesced(outputs, inputs, opts);
  for (const auto& tensor : outputs) {
    c10d::register_work(tensor, work);
  }
  return outputs;
}

at::Tensor reduce_scatter_tensor(
    const at::Tensor& input,
    std::string reduce_op,
    int64_t group_size,
    std::string group_name) {
  TORCH_CHECK(input.is_contiguous());
  if (input.is_complex()) {
    auto real_input = at::view_as_real(input);
    std::vector<at::Tensor> inputs{real_input};
    return at::view_as_complex(reduce_scatter_tensor_coalesced(
        inputs, std::move(reduce_op), group_size, std::move(group_name))[0]);
  }
  std::vector<at::Tensor> inputs{input};
  return reduce_scatter_tensor_coalesced(
      inputs, std::move(reduce_op), group_size, std::move(group_name))[0];
}

at::Tensor all_to_all_single(
    const at::Tensor& input,
    c10::SymIntArrayRef _output_split_sizes,
    c10::SymIntArrayRef _input_split_sizes,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name) {
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

  auto group = c10d::resolve_process_group(group_name);
  auto work = group->alltoall_base(
      output,
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<at::Tensor&>(input),
      output_split_sizes,
      input_split_sizes);
  c10d::register_work(output, work);
  return output;
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
at::Tensor& broadcast_(at::Tensor& input, int64_t src, std::string group_name) {
  c10d::BroadcastOptions opts;
  opts.rootRank = src;
  auto input_real = input.is_complex() ? at::view_as_real(input) : input;
  std::vector<at::Tensor> inputs{input_real};

  auto group = c10d::resolve_process_group(group_name);
  auto work = group->broadcast(inputs, opts);
  c10d::register_work(input, work);
  return input;
}

at::Tensor broadcast(
    const at::Tensor& input,
    int64_t src,
    std::string group_name) {
  auto output = input.clone(at::MemoryFormat::Contiguous);
  return broadcast_(output, src, std::move(group_name));
}

} // namespace c10d

TORCH_LIBRARY(_c10d_functional, m) {
  m.def(
      "all_reduce(Tensor input, str reduce_op, str group_name) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, c10d::all_reduce),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "all_reduce_(Tensor(a!) input, str reduce_op, str group_name) -> Tensor(a!)",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, c10d::all_reduce_),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "all_reduce_coalesced(Tensor[] inputs, str reduce_op, str group_name) -> Tensor[]",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          c10d::all_reduce_coalesced),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "all_reduce_coalesced_(Tensor[](a!) inputs, str reduce_op, str group_name) -> Tensor[](a!)",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          c10d::all_reduce_coalesced_),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "all_gather_into_tensor_out(Tensor input, int group_size, str group_name, *, Tensor(a!) out) -> Tensor(a!)",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          c10d::all_gather_into_tensor_out),
      {at::Tag::pt2_compliant_tag, at::Tag::needs_contiguous_strides});

  m.def(
      "all_gather_into_tensor(Tensor input, int group_size, str group_name) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          c10d::all_gather_into_tensor),
      {at::Tag::pt2_compliant_tag, at::Tag::needs_contiguous_strides});

  m.def(
      "all_gather_into_tensor_coalesced(Tensor[] inputs, int group_size, str group_name) -> Tensor[]",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          c10d::all_gather_into_tensor_coalesced),
      {at::Tag::pt2_compliant_tag, at::Tag::needs_contiguous_strides});

  m.def(
      "reduce_scatter_tensor(Tensor input, str reduce_op, int group_size, str group_name) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          c10d::reduce_scatter_tensor),
      {at::Tag::pt2_compliant_tag, at::Tag::needs_contiguous_strides});

  m.def(
      "reduce_scatter_tensor_coalesced(Tensor[] inputs, str reduce_op, int group_size, str group_name) -> Tensor[]",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          c10d::reduce_scatter_tensor_coalesced),
      {at::Tag::pt2_compliant_tag, at::Tag::needs_contiguous_strides});

  m.def(
      "all_to_all_single("
      "Tensor input, "
      "SymInt[] output_split_sizes, "
      "SymInt[] input_split_sizes, "
      "str group_name) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, c10d::all_to_all_single),
      {at::Tag::pt2_compliant_tag, at::Tag::needs_contiguous_strides});

  m.def(
      "broadcast(Tensor input, int src, str group_name) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, c10d::broadcast),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "broadcast_(Tensor(a!) input, int src, str group_name) -> Tensor(a!)",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, c10d::broadcast_),
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
      // NOLINTNEXTLINE(performance-unnecessary-value-param)
      std::string group_name) {
    // swap sizes for backwards pass
    ctx->saved_data["output_split_sizes"] = input_split_sizes.vec();
    ctx->saved_data["input_split_sizes"] = output_split_sizes.vec();
    ctx->saved_data["group_name"] = group_name;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("_c10d_functional::all_to_all_single", "")
        .typed<decltype(c10d::all_to_all_single)>()
        .call(input, output_split_sizes, input_split_sizes, group_name);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_out_list) {
    std::vector<c10::SymInt> output_split_sizes =
        ctx->saved_data["output_split_sizes"].toSymIntVector();
    std::vector<c10::SymInt> input_split_sizes =
        ctx->saved_data["input_split_sizes"].toSymIntVector();
    const std::string& group_name = ctx->saved_data["group_name"].toStringRef();

    DCHECK(grad_out_list.size() == 1);
    auto grad_out = grad_out_list[0].contiguous();

    auto out =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("_c10d_functional::all_to_all_single", "")
            .typed<decltype(c10d::all_to_all_single)>()
            .call(grad_out, output_split_sizes, input_split_sizes, group_name);

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
    const std::string& group_name) {
  return AllToAllSingle::apply(
      input, output_split_sizes, input_split_sizes, group_name);
}

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
        .typed<decltype(c10d::reduce_scatter_tensor)>()
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
            .typed<decltype(c10d::all_gather_into_tensor)>()
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
        .typed<decltype(c10d::all_gather_into_tensor)>()
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
            .typed<decltype(c10d::reduce_scatter_tensor)>()
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
      "str group_name) -> Tensor",
      torch::dispatch(c10::DispatchKey::Autograd, ::all_to_all_single_autograd),
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
    const std::string& group_name) {
  auto group = c10d::resolve_process_group(group_name);
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
} // namespace

// DTensor comm op registry
TORCH_LIBRARY(_dtensor, m) {
  m.def(
      "shard_dim_alltoall(Tensor input, int gather_dim, int shard_dim, str group_name) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, ::shard_dim_alltoall),
      {at::Tag::pt2_compliant_tag});
}
