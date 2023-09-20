#include <string>

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/DispatchKey.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

// _c10d_functional ops and relevant runtime APIs.
//
// _c10d_functional is a native port of c10d_functional which currently resides
// in python. _c10d_functional will replace c10d_functional once functionality
// parity is verified.
//
// Motivations for porting c10d_functional ops to c++:
// - Support collective ops for aot inductor and native interpreters
// - Support multi-threaded native runtimes
// - Unify collective codegens between inductor python and aot inductor
//
// Changes compared to c10d_functional:
// - The namespace now contains in-place variants for collective calls inductor
// decides to de-functionalize (should we put them in a different namespace?).
// - Process group resolution now only relies on tag. Ranks are useful for the
// compiler to determine whether two collectives using different process groups
// are fuse-able. However, they are not required for process group resolution.
// - Process group resolution now doesn't upsert a process group.

static thread_local std::unordered_map<void*, c10::intrusive_ptr<c10d::Work>>
    pendingWork;

namespace {

const std::unordered_map<std::string, c10d::ReduceOp> strToReduceOp = {
    {"sum", c10d::ReduceOp(c10d::ReduceOp::RedOpType::SUM)},
    {"avg", c10d::ReduceOp(c10d::ReduceOp::RedOpType::AVG)},
    {"product", c10d::ReduceOp(c10d::ReduceOp::RedOpType::PRODUCT)},
    {"min", c10d::ReduceOp(c10d::ReduceOp::RedOpType::MIN)},
    {"max", c10d::ReduceOp(c10d::ReduceOp::RedOpType::MAX)},
    {"band", c10d::ReduceOp(c10d::ReduceOp::RedOpType::BAND)},
    {"bor", c10d::ReduceOp(c10d::ReduceOp::RedOpType::BOR)},
    {"bxor", c10d::ReduceOp(c10d::ReduceOp::RedOpType::BXOR)},
    // {"premul_sum", c10d::ReduceOp(c10d::ReduceOp::RedOpType::PREMUL_SUM)},
    {"unused", c10d::ReduceOp(c10d::ReduceOp::RedOpType::UNUSED)}};

at::Tensor all_reduce_(
    at::Tensor input,
    const std::string& reduceOp,
    const std::string& tag,
    const std::vector<int64_t> ranks,
    const int64_t groupSize) {
  c10d::AllreduceOptions opts;
  auto reduceOpIt = strToReduceOp.find(reduceOp);
  if (reduceOpIt == strToReduceOp.end()) {
    LOG(FATAL) << "Unrecognized reduceOp: " << reduceOp;
  }
  opts.reduceOp = reduceOpIt->second;

  std::vector<at::Tensor> inputs{input};
  auto pg = c10d::ProcessGroup::resolveFromName(tag);
  if (groupSize != pg->getSize()) {
    LOG(FATAL) << "Mismatch between group size argument and world size"
                  "of process group ("
               << tag << ")";
  }
  auto work = pg->allreduce(inputs, opts);
  pendingWork[input.data_ptr()] = work;
  return input;
}

at::Tensor all_reduce(
    const at::Tensor& input,
    const std::string& reduceOp,
    const std::string& tag,
    const std::vector<int64_t> ranks,
    const int64_t groupSize) {
  auto output = input.clone();
  return all_reduce_(output, reduceOp, tag, ranks, groupSize);
}

std::vector<at::Tensor> all_reduce_coalesced_(
    std::vector<at::Tensor> inputs,
    const std::string& reduceOp,
    const std::string& tag,
    const std::vector<int64_t> ranks,
    const int64_t groupSize) {
  c10d::AllreduceCoalescedOptions opts;
  auto reduceOpIt = strToReduceOp.find(reduceOp);
  if (reduceOpIt == strToReduceOp.end()) {
    LOG(FATAL) << "Unrecognized reduceOp: " << reduceOp;
  }
  opts.reduceOp = reduceOpIt->second;

  auto pg = c10d::ProcessGroup::resolveFromName(tag);
  if (groupSize != pg->getSize()) {
    LOG(FATAL) << "Mismatch between group size argument and world size"
                  "of process group ("
               << tag << ")";
  }
  auto work = pg->allreduce_coalesced(inputs, opts);
  for (const auto& tensor : inputs) {
    // c10d::Work::wait is idempotent
    pendingWork[tensor.data_ptr()] = work;
  }
  return inputs;
}

std::vector<at::Tensor> all_reduce_coalesced(
    const std::vector<at::Tensor>& inputs,
    const std::string& reduceOp,
    const std::string& tag,
    const std::vector<int64_t> ranks,
    const int64_t groupSize) {
  std::vector<at::Tensor> outputs;
  for (const auto& tensor : inputs) {
    outputs.push_back(tensor.clone());
  }
  return all_reduce_coalesced_(outputs, reduceOp, tag, ranks, groupSize);
}

at::Tensor allocateAllGatherOutput(const at::Tensor& input, int64_t groupSize) {
  auto outputSize = input.sizes().vec();
  outputSize[0] *= groupSize;
  return at::empty(
      outputSize,
      at::TensorOptions().dtype(input.dtype()).device(input.device()));
}

std::vector<at::Tensor> all_gather_into_tensor_coalesced(
    const std::vector<at::Tensor>& inputs,
    const std::string& tag,
    const std::vector<int64_t> ranks,
    const int64_t groupSize) {
  std::vector<at::Tensor> outputs;
  for (const auto& tensor : inputs) {
    outputs.push_back(allocateAllGatherOutput(tensor, groupSize));
  }
  // TODO: assert groupSize
  auto pg = c10d::ProcessGroup::resolveFromName(tag);
  if (groupSize != pg->getSize()) {
    LOG(FATAL) << "Mismatch between group size argument and world size"
                  "of process group ("
               << tag << ")";
  }
  auto work = pg->allgather_into_tensor_coalesced(
      outputs, const_cast<std::vector<at::Tensor>&>(inputs));
  for (const auto& tensor : outputs) {
    pendingWork[tensor.data_ptr()] = work;
  }
  return outputs;
}

at::Tensor all_gather_into_tensor(
    const at::Tensor& input,
    const std::string& tag,
    const std::vector<int64_t> ranks,
    const int64_t groupSize) {
  std::vector<at::Tensor> inputs{input};
  return all_gather_into_tensor_coalesced(inputs, tag, ranks, groupSize)[0];
}

at::Tensor allocateReduceScatterOutput(
    const at::Tensor& input,
    int64_t groupSize) {
  auto outputSize = input.sizes().vec();
  outputSize[0] /= groupSize;
  return at::empty(
      outputSize,
      at::TensorOptions().dtype(input.dtype()).device(input.device()));
}

std::vector<at::Tensor> reduce_scatter_tensor_coalesced(
    const std::vector<at::Tensor>& inputs,
    const std::string& reduceOp,
    const std::string& tag,
    const std::vector<int64_t>& ranks,
    const int64_t groupSize) {
  c10d::ReduceScatterOptions opts;
  auto reduceOpIt = strToReduceOp.find(reduceOp);
  if (reduceOpIt == strToReduceOp.end()) {
    LOG(FATAL) << "Unrecognized reduceOp: " << reduceOp;
  }
  opts.reduceOp = reduceOpIt->second;

  std::vector<at::Tensor> outputs;
  for (const auto& tensor : inputs) {
    outputs.push_back(allocateReduceScatterOutput(tensor, groupSize));
  }
  auto pg = c10d::ProcessGroup::resolveFromName(tag);
  if (groupSize != pg->getSize()) {
    LOG(FATAL) << "Mismatch between group size argument and world size"
                  "of process group ("
               << tag << ")";
  }
  auto work = pg->reduce_scatter_tensor_coalesced(
      outputs, const_cast<std::vector<at::Tensor>&>(inputs), opts);
  for (const auto& tensor : outputs) {
    // c10d::Work::wait is idempotent
    pendingWork[tensor.data_ptr()] = work;
  }
  return outputs;
}

at::Tensor reduce_scatter_tensor(
    at::Tensor input,
    const std::string& reduceOp,
    const std::string& tag,
    const std::vector<int64_t>& ranks,
    const int64_t groupSize) {
  std::vector<at::Tensor> inputs{input};
  return reduce_scatter_tensor_coalesced(
      inputs, reduceOp, tag, ranks, groupSize)[0];
}

at::Tensor wait_tensor(const at::Tensor& tensor) {
  auto it = pendingWork.find(tensor.data_ptr());
  if (it == pendingWork.end()) {
    LOG(FATAL)
        << "No pending collective is associated with the input tensor. "
           "This typically means that the input tensor is not a collective output, "
           "or the tensor has already been waited on.";
  }
  it->second->wait();
  pendingWork.erase(it);
  return tensor;
}

} // namespace

TORCH_LIBRARY(_c10d_functional, m) {
  // TODO(yifu): is it neccessary to keep the same argument names used in
  // c10d_functional? It would be nice to modify it to be more consistent.
  m.def(
      "all_reduce(Tensor self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, ::all_reduce));

  m.def(
      "all_reduce_(Tensor self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, ::all_reduce_));

  m.def(
      "all_reduce_coalesced(Tensor[] self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, ::all_reduce_coalesced));

  m.def(
      "all_reduce_coalesced_(Tensor[] self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          ::all_reduce_coalesced_));

  m.def(
      "all_gather_into_tensor(Tensor shard, str tag, int[] ranks, int group_size) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          ::all_gather_into_tensor));

  m.def(
      "all_gather_into_tensor_coalesced(Tensor[] input, str tag, int[] ranks, int group_size) -> Tensor[]",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          ::all_gather_into_tensor_coalesced));

  m.def(
      "reduce_scatter_tensor(Tensor input, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          ::reduce_scatter_tensor));

  m.def(
      "reduce_scatter_tensor_coalesced(Tensor[] inputs, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          ::reduce_scatter_tensor_coalesced));

  m.def(
      "wait_tensor(Tensor self) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, ::wait_tensor));
}
