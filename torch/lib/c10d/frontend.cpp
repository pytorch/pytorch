#include <c10d/frontend.hpp>

#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>
#include <c10/util/Exception.h>

#include <chrono>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace c10d {

namespace {

void maybePreprocessComplexTensor(at::Tensor& tensor) {
  if(!tensor.is_complex()) {
    return;
  }

  tensor = at::view_as_real(tensor);
}

void maybePreprocessComplexTensor(std::vector<at::Tensor>& tensors) {
  for(at::Tensor& t : tensors) {
    maybePreprocessComplexTensor(t);
  }
}

void maybePreprocessComplexTensor(std::vector<std::vector<at::Tensor>>& tensors_lists) {
  for(std::vector<at::Tensor>& t : tensors_lists) {
    maybePreprocessComplexTensor(tensors_lists);
  }
}

void assertReduceOpSupportsComplexTensor(ReduceOp op) {
  switch (op) {
    case ReduceOp::MAX:
    case ReduceOp::MIN:
    case ReduceOp::PRODUCT:
      TORCH_CHECK(
          false,
          "all_reduce does not support requested Reduce op on complex tensors");
    default:
      return;
  }
}

}  // namespace anonymous

std::string Backend::get(const std::string& backend_type) {
	return backend_type;
}

void Backend::registerBackend() {
	TORCH_CHECK(false, "Registering third-party backend is currently not supported by TorchScript-friendly c10d");
}

// Note: We assume that group.WORLD equates default_pg_. Otherwise,
// we need many additional conditionals to check whether group is WORLD and
// then use default_pg_ explicitly.

int64_t DistributedC10d::getRank(
    const c10::intrusive_ptr<ProcessGroup>& group) const {
  if (rankNotInGroup(group)) {
    return -1;
  }

  return group->getRank();
}

int64_t DistributedC10d::getWorldSize(
    const c10::intrusive_ptr<ProcessGroup>& group) const {
  if (rankNotInGroup(group)) {
    return -1;
  }

  return getGroupSize(group);
}

int64_t DistributedC10d::getGroupSize(
    const c10::intrusive_ptr<ProcessGroup>& group) const {
  if (group == default_pg_) {
    default_pg_->getSize();
  }

  auto it = pg_group_ranks_.find(group);
  TORCH_CHECK(it != pg_group_ranks_.end(), "The given group does not exist");

  return it->second.size();
}

void DistributedC10d::checkDefaultPg() const {
  TORCH_CHECK(default_pg_, "Default process group is not initialized");
}

c10::intrusive_ptr<ProcessGroup> DistributedC10d::worldProcessGroup() {
  checkDefaultPg();
  return default_pg_;
}

bool DistributedC10d::rankNotInGroup(
    const c10::intrusive_ptr<ProcessGroup>& group) const {
  if (group == default_pg_) {
    return false;
  }
  return group;
}

int64_t DistributedC10d::getGroupRank(
    const c10::intrusive_ptr<ProcessGroup>& group,
    const int64_t rank) const {
  TORCH_CHECK(
      group != default_pg_,
      "group.WORLD does not have local rank to global rank mapping");

  auto it = pg_group_ranks_.find(group);
  TORCH_CHECK(it != pg_group_ranks_.end(), "The given group does not exist");

  auto& group_rank_map = it->second;
  auto g_it = group_rank_map.find(rank);
  if (g_it == group_rank_map.end()) {
    std::string group_name = "Unknown";
    auto name_it = pg_names_.find(group);
    if (name_it != pg_names_.end()) {
      group_name = name_it->second;
    }

    TORCH_CHECK(
        false,
        "The global rank ",
        rank,
        " is not part of the group ",
        group_name);
  }

  return g_it->second;
}

int64_t DistributedC10d::getGlobalRank(
    const c10::intrusive_ptr<ProcessGroup>& group,
    const int64_t group_rank) const {
  TORCH_CHECK(
      group != default_pg_,
      "group.WORLD does not have local rank to global rank mapping");

  auto it = pg_group_ranks_.find(group);
  TORCH_CHECK(it != pg_group_ranks_.end(), "The given group does not exist");

  auto& group_rank_map = it->second;
  for (const auto& p : group_rank_map) {
    if (p.second == group_rank) {
      return p.first;
    }
  }

  TORCH_CHECK(false, "The group rank is not part of the group");
}

std::string DistributedC10d::getBackend(
    const c10::intrusive_ptr<ProcessGroup>& group) {
  TORCH_CHECK(!rankNotInGroup(group), "Invalid process group specified");

  auto it = pg_map_.find(group);
  TORCH_CHECK(it != pg_map_.end(), "The given group does not exist");

  return it->second.first;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::isend(
    at::Tensor tensor,
    int64_t dst,
    const c10::intrusive_ptr<ProcessGroup>& group,
    c10::optional<int64_t>& tag) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  std::vector<at::Tensor> inputs = {std::move(tensor)};

  checkDefaultPg();
  if (group == default_pg_) {
    return default_pg_->send(inputs, dst, tag.value_or(0));
  }

  auto group_dst_rank = getGroupRank(group, dst);
  return group->send(inputs, group_dst_rank, tag.value_or(0));
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::irecv(
    at::Tensor tensor,
    int64_t src,
    const c10::intrusive_ptr<ProcessGroup>& group,
    c10::optional<int64_t>& tag) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  std::vector<at::Tensor> inputs = {std::move(tensor)};

  checkDefaultPg();
  if (group == default_pg_) {
    return default_pg_->recv(inputs, src, tag.value_or(0));
  }

  auto group_dst_rank = getGroupRank(group, src);
  return group->recv(inputs, group_dst_rank, tag.value_or(0));
}

void DistributedC10d::send(
    at::Tensor tensor,
    int64_t dst,
    const c10::intrusive_ptr<ProcessGroup>& group,
    c10::optional<int64_t>& tag) {
  auto work = isend(std::move(tensor), dst, group, tag);
  if (work) {
    work->wait();
  }
}

int64_t DistributedC10d::recv(
    at::Tensor tensor,
    const c10::optional<int64_t>& src,
    const c10::intrusive_ptr<ProcessGroup>& group,
    c10::optional<int64_t>& tag) {
  if (rankNotInGroup(group)) {
    return -1;
  }

  std::vector<at::Tensor> outputs = {std::move(tensor)};
  if (!src.has_value()) {
    auto work = group->recvAnysource(outputs, tag.value_or(0));
    work->wait();
    auto src_rank = work->sourceRank();
    if (group == default_pg_) {
      return src_rank;
    }

    return getGlobalRank(group, src_rank);
  }

  if (group == default_pg_) {
    group->recv(outputs, src.value(), tag.value_or(0))->wait();
  } else {
    int64_t group_src_rank = getGroupRank(group, src.value());
    group->recv(outputs, group_src_rank, tag.value_or(0))->wait();
  }

  return src.value();
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::broadcastMultiGPU(
    std::vector<at::Tensor>& tensor_list,
    int64_t src,
    const c10::intrusive_ptr<ProcessGroup>& group,
    bool async_op,
    int64_t src_tensor) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  BroadcastOptions opts;
  opts.rootRank = src;
  opts.rootTensor = src_tensor;

  checkDefaultPg();
  c10::intrusive_ptr<ProcessGroup::Work> work;
  if (group == default_pg_) {
    work = default_pg_->broadcast(tensor_list, opts);
  } else {
    int64_t group_src_rank = getGroupRank(group, src);
    opts.rootRank = group_src_rank;
    work = group->broadcast(tensor_list, opts);
  }

  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::broadcast(
    at::Tensor tensor,
    int64_t src,
    const c10::intrusive_ptr<ProcessGroup>& group,
    bool async_op) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  BroadcastOptions opts;
  opts.rootRank = src;
  opts.rootTensor = 0;

  std::vector<at::Tensor> tensors = {std::move(tensor)};
  c10::intrusive_ptr<ProcessGroup::Work> work;
  checkDefaultPg();
  if (group == default_pg_) {
    work = group->broadcast(tensors, opts);
  } else {
    int64_t group_src_rank = getGroupRank(group, src);
    opts.rootRank = group_src_rank;
    work = group->broadcast(tensors, opts);
  }

  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::allReduceMultiGPU(
    std::vector<at::Tensor>& tensor_list,
    const c10::intrusive_ptr<ProcessGroup>& group,
    ReduceOp op,
    bool async_op) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  AllreduceOptions opts;
  opts.reduceOp = op;

  assertReduceOpSupportsComplexTensor(op);
  maybePreprocessComplexTensor(tensor_list);

  auto work = group->allreduce(tensor_list, opts);
  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::allReduce(
    at::Tensor tensor,
    const c10::intrusive_ptr<ProcessGroup>& group,
    ReduceOp op,
    bool async_op) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  AllreduceOptions opts;
  opts.reduceOp = op;

  assertReduceOpSupportsComplexTensor(op);
  maybePreprocessComplexTensor(tensor);

  std::vector<at::Tensor> tensors = {std::move(tensor)};
  auto work = group->allreduce(tensors, opts);
  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::allReduceCoalesced(
    std::vector<at::Tensor>& tensors,
    const c10::intrusive_ptr<ProcessGroup>& group,
    ReduceOp op,
    bool async_op) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  AllreduceCoalescedOptions opts;
  opts.reduceOp = op;

  assertReduceOpSupportsComplexTensor(op);
  maybePreprocessComplexTensor(tensors);

  auto work = group->allreduce_coalesced(tensors, opts);
  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::reduceMultiGPU(
    std::vector<at::Tensor>& tensor_list,
    int64_t dst,
    const c10::intrusive_ptr<ProcessGroup>& group,
    ReduceOp op,
    bool async_op,
    int64_t dst_tensor) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  ReduceOptions opts;
  opts.reduceOp = op;
  opts.rootRank = dst;
  opts.rootTensor = dst_tensor;

  checkDefaultPg();

  c10::intrusive_ptr<ProcessGroup::Work> work;
  if (group == default_pg_) {
    work = group->reduce(tensor_list, opts);
  } else {
    int64_t group_dst_rank = getGroupRank(group, dst);
    opts.rootRank = group_dst_rank;
    work = group->reduce(tensor_list, opts);
  }

  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::reduce(
    at::Tensor tensor,
    int64_t dst,
    const c10::intrusive_ptr<ProcessGroup>& group,
    ReduceOp op,
    bool async_op) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  ReduceOptions opts;
  opts.reduceOp = op;
  opts.rootRank = dst;

  checkDefaultPg();
  c10::intrusive_ptr<ProcessGroup::Work> work;
  std::vector<at::Tensor> tensors = {std::move(tensor)};
  if (group == default_pg_) {
    work = group->reduce(tensors, opts);
  } else {
    int64_t group_dst_rank = getGroupRank(group, dst);
    opts.rootRank = group_dst_rank;
    work = group->reduce(tensors, opts);
  }

  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::allGatherMultiGPU(
    std::vector<std::vector<at::Tensor>>& output_tensor_lists,
    std::vector<at::Tensor>& input_tensor_list,
    const c10::intrusive_ptr<ProcessGroup>& group,
    bool async_op) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  maybePreprocessComplexTensor(output_tensor_lists);
  maybePreprocessComplexTensor(input_tensor_list);

  auto work = group->allgather(output_tensor_lists, input_tensor_list);

  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::allGather(
    std::vector<at::Tensor>& tensor_list,
    at::Tensor tensor,
    const c10::intrusive_ptr<ProcessGroup>& group,
    bool async_op) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  maybePreprocessComplexTensor(tensor_list);
  maybePreprocessComplexTensor(tensor);

  std::vector<std::vector<at::Tensor>> output_tensor_lists = {std::move(tensor_list)};
  std::vector<at::Tensor> input_tensor_list = {std::move(tensor)};
  auto work = group->allgather(output_tensor_lists, input_tensor_list);

  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::allGatherCoalesced(
    std::vector<std::vector<at::Tensor>>& output_tensor_lists,
    std::vector<at::Tensor>& input_tensor_list,
    const c10::intrusive_ptr<ProcessGroup>& group,
    bool async_op) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  maybePreprocessComplexTensor(output_tensor_lists);
  maybePreprocessComplexTensor(input_tensor_list);

  auto work =
      group->allgather_coalesced(output_tensor_lists, input_tensor_list);

  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::gather(
    at::Tensor tensor,
    const c10::optional<std::vector<at::Tensor>>& gather_list,
    const c10::intrusive_ptr<ProcessGroup>& group,
    int64_t dst,
    bool async_op) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  auto my_rank = group->getRank();

  std::vector<std::vector<at::Tensor>> output_tensors;

  if (dst == my_rank) {
    TORCH_CHECK(
        gather_list.has_value(),
        "Argument ``gather_list`` must be specified on destination rank");
    output_tensors.push_back(gather_list.value());
  } else {
    TORCH_CHECK(
        !gather_list.has_value(),
        "Argument ``gather_list`` must NOT be specified on non-destination ranks.");
  }

  std::vector<at::Tensor> input_tensors = {std::move(tensor)};

  GatherOptions opts;
  opts.rootRank = dst;

  c10::intrusive_ptr<ProcessGroup::Work> work;
  if (group == default_pg_) {
    work = group->gather(output_tensors, input_tensors, opts);
  } else {
    int64_t group_dst_rank = getGroupRank(group, dst);
    opts.rootRank = group_dst_rank;
    work = group->gather(output_tensors, input_tensors, opts);
  }

  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::scatter(
    at::Tensor tensor,
    std::vector<at::Tensor>& scatter_list,
    const c10::intrusive_ptr<ProcessGroup>& group,
    int64_t src,
    bool async_op) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  auto my_rank = getRank(default_pg_);

  std::vector<at::Tensor> output_tensors = {std::move(tensor)};
  std::vector<std::vector<at::Tensor>> input_tensors;
  if (src == my_rank) {
    input_tensors.push_back(scatter_list);
  }

  ScatterOptions opts;
  opts.rootRank = src;

  c10::intrusive_ptr<ProcessGroup::Work> work;
  if (group == default_pg_) {
    work = group->scatter(output_tensors, input_tensors, opts);
  } else {
    int64_t group_src_rank = getGroupRank(group, src);
    opts.rootRank = group_src_rank;
    work = group->scatter(output_tensors, input_tensors, opts);
  }

  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::reduceScatterMultiGPU(
    std::vector<at::Tensor>& output_tensor_list,
    std::vector<std::vector<at::Tensor>>& input_tensor_lists,
    const c10::intrusive_ptr<ProcessGroup>& group,
    ReduceOp op,
    bool async_op) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  ReduceScatterOptions opts;
  opts.reduceOp = op;

  auto work =
      group->reduce_scatter(output_tensor_list, input_tensor_lists, opts);

  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::reduceScatter(
    at::Tensor output,
    std::vector<at::Tensor>& input_tensor_list,
    const c10::intrusive_ptr<ProcessGroup>& group,
    ReduceOp op,
    bool async_op) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  ReduceScatterOptions opts;
  opts.reduceOp = op;

  std::vector<at::Tensor> output_tensor_list = {std::move(output)};
  std::vector<std::vector<at::Tensor>> input_tensor_lists = {std::move(input_tensor_list)};

  auto work =
      group->reduce_scatter(output_tensor_list, input_tensor_lists, opts);

  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::allToAllSingle(
    at::Tensor output,
    at::Tensor input,
    std::vector<int64_t>& output_split_sizes,
    std::vector<int64_t>& input_split_sizes,
    const c10::intrusive_ptr<ProcessGroup>& group,
    bool async_op) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  AllToAllOptions opts;
  auto work = group->alltoall_base(
      output, input, output_split_sizes, input_split_sizes, opts);

  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::allToAll(
    std::vector<at::Tensor>& output_tensor_list,
    std::vector<at::Tensor>& input_tensor_list,
    const c10::intrusive_ptr<ProcessGroup>& group,
    bool async_op) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  AllToAllOptions opts;
  auto work = group->alltoall(output_tensor_list, input_tensor_list, opts);

  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

c10::intrusive_ptr<ProcessGroup::Work> DistributedC10d::barrier(
    const c10::intrusive_ptr<ProcessGroup>& group,
    bool async_op) {
  c10::intrusive_ptr<ProcessGroup::Work> empty_work;
  if (rankNotInGroup(group)) {
    return empty_work;
  }

  auto work = group->barrier();

  if (async_op) {
    return work;
  }
  work->wait();
  return empty_work;
}

} // namespace c10d
