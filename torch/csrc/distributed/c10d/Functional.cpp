#include <torch/csrc/distributed/c10d/Functional.hpp>

#include <shared_mutex>

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/DispatchKey.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace {

// NOTE [Process Group Resolution]
//
// c10d_functional requires process group objects to be associated with a tag.
// Using this tag, the collective ops can determine the appropriate process
// group object to use. The mechanism that maintains the association must
// accommodate both the one-rank-per-process and one-rank-per-thread scenarios.
// Specifically, in one-rank-per-thread scenario, different thread should
// associate with different process group objects based on the same tag, making
// the resolution logic context-dependent. While using thread_local might
// appear as a feasible solution, it falls short during backward passes. This
// is because an autograd thread cannot be reliably linked to a specific rank.
//
// The solution is to use fwd_thread_id as the context identifier. For
// non-autograd threads, fwd_thread_id is simply the thread's id. For autograd
// threads, fwd_thread_id the the id of the thread that created the current
// node.
uint64_t get_fwd_thread_id() {
  const auto node = torch::autograd::get_current_node();
  return node == nullptr ? at::RecordFunction::currentThreadId()
                         : node->thread_id();
}

class WorkRegistry {
 public:
  void register_work(
      const at::Tensor& tensor,
      c10::intrusive_ptr<c10d::Work> work) {
    const auto storage = tensor.storage().getWeakStorageImpl();
    std::unique_lock lock(lock_);
    const auto it = registry_.find(storage);
    TORCH_CHECK(
        it == registry_.end() || it->second != work,
        "The tensor storage is already associated with another work.");
    registry_[storage] = work;
  }

  void pop_and_wait_work(const at::Tensor& tensor) {
    const auto storage = tensor.storage().getWeakStorageImpl();
    std::unique_lock lock(lock_);
    auto it = registry_.find(storage);
    TORCH_CHECK(
        it != registry_.end(),
        "No pending collective is associated with the tensor storage. "
        "This typically means that the tensor is not a collective output, "
        "or the tensor has already been waited on.");
    auto work = it->second;
    registry_.erase(it);
    lock.unlock();
    work->wait();
  }

  static WorkRegistry& get() {
    std::shared_lock read_lock(work_registry_table_lock);
    return work_registry_table[get_fwd_thread_id()];
  }

  static void maybe_init_for_current_thread() {
    std::unique_lock write_lock(work_registry_table_lock);
    work_registry_table.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(get_fwd_thread_id()),
        std::forward_as_tuple());
  }

 private:
  std::unordered_map<
      c10::weak_intrusive_ptr<c10::StorageImpl>,
      c10::intrusive_ptr<c10d::Work>>
      registry_;
  // The lock protects registry_ which can be concurrently accessed by
  // different autograd threads. For GPU use cases, this will not lead to
  // contention since each device is driven by a single autograd thread.
  std::mutex lock_;

  // Maintain a WorkRegistry for each fwd_thread_id. This minimizes contention
  // while allowing (1) collectives issued in non-autograd threads to be waited
  // by autograd threads (2) collectives issued in an autograd thread to be
  // waited by another autograd thread.
  static std::unordered_map<uint64_t, WorkRegistry> work_registry_table;
  // The lock protects the initialization of work_registry_table and will not
  // lead to contention during collective op invocations.
  static std::shared_mutex work_registry_table_lock;
};

std::unordered_map<uint64_t, WorkRegistry> WorkRegistry::work_registry_table;
std::shared_mutex WorkRegistry::work_registry_table_lock;

std::map<
    std::pair<std::string, uint64_t>,
    c10::weak_intrusive_ptr<c10d::ProcessGroup>>
    pg_registry;
// The lock protects the initialization of pg_registry and will not lead to
// contention during collective op invocations.
std::shared_mutex pg_registry_lock;

} // namespace

namespace c10d_functional {

void register_process_group(
    const std::string& tag,
    c10::intrusive_ptr<c10d::ProcessGroup> pg) {
  const auto fwd_thread_id = get_fwd_thread_id();
  TORCH_CHECK(
      fwd_thread_id == at::RecordFunction::currentThreadId(),
      "register_process_group cannot be called in a autograd thread.");

  {
    std::unique_lock write_lock(pg_registry_lock);
    auto it = pg_registry.find(std::make_pair(tag, fwd_thread_id));
    TORCH_CHECK(
        it == pg_registry.end(),
        "A process group is already registered for tag ",
        tag);
    pg_registry.emplace(std::make_pair(tag, fwd_thread_id), pg);
  }

  WorkRegistry::maybe_init_for_current_thread();
}

c10::intrusive_ptr<c10d::ProcessGroup> resolve_process_group(
    const std::string& tag) {
  const auto fwd_thread_id = get_fwd_thread_id();
  std::shared_lock lock(pg_registry_lock);
  auto it = pg_registry.find(std::make_pair(tag, fwd_thread_id));
  TORCH_CHECK(
      it != pg_registry.end(),
      "Could not resolve the process group registered for tag ",
      tag);

  auto pg = it->second.lock();
  TORCH_CHECK(
      pg != nullptr,
      "Process group registered for tag ",
      tag,
      " has already been destroyed.");
  return pg;
}

} // namespace c10d_functional

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

at::Tensor all_reduce_(
    at::Tensor input,
    const std::string& reduce_op,
    const std::string& tag) {
  c10d::AllreduceOptions opts;
  opts.reduceOp = to_reduce_op(reduce_op);

  std::vector<at::Tensor> inputs{input};
  auto pg = c10d_functional::resolve_process_group(tag);
  auto work = pg->allreduce(inputs, opts);
  WorkRegistry::get().register_work(input, work);
  return input;
}

at::Tensor all_reduce(
    const at::Tensor& input,
    const std::string& reduce_op,
    const std::string& tag) {
  auto output = input.clone();
  return all_reduce_(output, reduce_op, tag);
}

std::vector<at::Tensor> all_reduce_coalesced_(
    std::vector<at::Tensor> inputs,
    const std::string& reduce_op,
    const std::string& tag) {
  c10d::AllreduceCoalescedOptions opts;
  opts.reduceOp = to_reduce_op(reduce_op);

  auto pg = c10d_functional::resolve_process_group(tag);
  auto work = pg->allreduce_coalesced(inputs, opts);
  for (const auto& tensor : inputs) {
    WorkRegistry::get().register_work(tensor, work);
  }
  return inputs;
}

std::vector<at::Tensor> all_reduce_coalesced(
    const std::vector<at::Tensor>& inputs,
    const std::string& reduce_op,
    const std::string& tag) {
  std::vector<at::Tensor> outputs;
  for (const auto& tensor : inputs) {
    outputs.push_back(tensor.clone());
  }
  return all_reduce_coalesced_(outputs, reduce_op, tag);
}

at::Tensor allocate_all_gather_output(
    const at::Tensor& input,
    int64_t group_size) {
  auto output_size = input.sizes().vec();
  output_size[0] *= group_size;
  return at::empty(
      output_size,
      at::TensorOptions().dtype(input.dtype()).device(input.device()));
}

std::vector<at::Tensor> all_gather_into_tensor_coalesced(
    const std::vector<at::Tensor>& inputs,
    const int64_t group_size,
    const std::string& tag) {
  std::vector<at::Tensor> outputs;
  for (const auto& tensor : inputs) {
    outputs.push_back(allocate_all_gather_output(tensor, group_size));
  }

  auto pg = c10d_functional::resolve_process_group(tag);
  auto work = pg->allgather_into_tensor_coalesced(
      outputs, const_cast<std::vector<at::Tensor>&>(inputs));
  for (const auto& tensor : outputs) {
    WorkRegistry::get().register_work(tensor, work);
  }
  return outputs;
}

at::Tensor all_gather_into_tensor(
    const at::Tensor& input,
    const int64_t group_size,
    const std::string& tag) {
  std::vector<at::Tensor> inputs{input};
  return all_gather_into_tensor_coalesced(inputs, group_size, tag)[0];
}

at::Tensor allocate_reduce_scatter_output(
    const at::Tensor& input,
    const int64_t group_size) {
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

std::vector<at::Tensor> reduce_scatter_tensor_coalesced(
    const std::vector<at::Tensor>& inputs,
    const std::string& reduce_op,
    const int64_t group_size,
    const std::string& tag) {
  c10d::ReduceScatterOptions opts;
  opts.reduceOp = to_reduce_op(reduce_op);
  std::vector<at::Tensor> outputs;
  for (const auto& tensor : inputs) {
    outputs.push_back(allocate_reduce_scatter_output(tensor, group_size));
  }

  auto pg = c10d_functional::resolve_process_group(tag);
  auto work = pg->reduce_scatter_tensor_coalesced(
      outputs, const_cast<std::vector<at::Tensor>&>(inputs), opts);
  for (const auto& tensor : outputs) {
    WorkRegistry::get().register_work(tensor, work);
  }
  return outputs;
}

at::Tensor reduce_scatter_tensor(
    at::Tensor input,
    const std::string& reduce_op,
    const int64_t group_size,
    const std::string& tag) {
  std::vector<at::Tensor> inputs{input};
  return reduce_scatter_tensor_coalesced(inputs, reduce_op, group_size, tag)[0];
}

at::Tensor wait_tensor(const at::Tensor& tensor) {
  WorkRegistry::get().pop_and_wait_work(tensor);
  return tensor;
}

} // namespace

TORCH_LIBRARY(_c10d_functional, m) {
  m.def(
      "all_reduce(Tensor input, str reduce_op, str tag) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, ::all_reduce));

  m.def(
      "all_reduce_(Tensor(a!) input, str reduce_op, str tag) -> Tensor(a!)",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, ::all_reduce_));

  m.def(
      "all_reduce_coalesced(Tensor[] inputs, str reduce_op, str tag) -> Tensor[]",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, ::all_reduce_coalesced));

  m.def(
      "all_reduce_coalesced_(Tensor[](a!) inputs, str reduce_op, str tag) -> Tensor[](a!)",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          ::all_reduce_coalesced_));

  m.def(
      "all_gather_into_tensor(Tensor input, int group_size, str tag) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          ::all_gather_into_tensor));

  m.def(
      "all_gather_into_tensor_coalesced(Tensor[] inputs, int group_size, str tag) -> Tensor[]",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          ::all_gather_into_tensor_coalesced));

  m.def(
      "reduce_scatter_tensor(Tensor input, str reduce_op, int group_size, str tag) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          ::reduce_scatter_tensor));

  m.def(
      "reduce_scatter_tensor_coalesced(Tensor[] inputs, str reduce_op, int group_size, str tag) -> Tensor[]",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd,
          ::reduce_scatter_tensor_coalesced));

  m.def(
      "wait_tensor(Tensor self) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, ::wait_tensor));
}
