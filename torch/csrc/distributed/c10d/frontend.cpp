#include <c10d/frontend.hpp>

#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>
#include <c10/util/Exception.h>
#include <c10d/PrefixStore.hpp>
#include <c10d/FileStore.hpp>
#include <c10d/TCPStore.hpp>
#include <c10d/Utils.hpp>

#include <chrono>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#ifdef USE_C10D_GLOO
#include <c10d/ProcessGroupGloo.hpp>
#endif

#ifdef USE_C10D_NCCL
#include <c10d/ProcessGroupNCCL.hpp>
#endif

#ifdef USE_C10D_MPI
#include <c10d/ProcessGroupMPI.hpp>
#endif

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
    maybePreprocessComplexTensor(t);
  }
}

void assertReduceOpSupportsComplexTensor(ReduceOp op) {
  switch (op) {
    case ReduceOp::MAX:
    case ReduceOp::MIN:
    case ReduceOp::PRODUCT:
      AT_ERROR(
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

c10::intrusive_ptr<DistributedC10d> DistributedC10d::get() {
  static c10::intrusive_ptr<DistributedC10d> singleton =
      c10::make_intrusive<DistributedC10d>();

  return singleton;
}

c10::intrusive_ptr<ProcessGroup> DistributedC10d::getProcessGroupByName(const std::string& name) const {
  auto it = std::find_if(
      pg_names_.begin(),
      pg_names_.end(),
      [&](const std::pair<c10::intrusive_ptr<ProcessGroup>, std::string>&
              pg_name) { return pg_name.second == name; });

  if (it == pg_names_.end()) {
    std::stringstream error;
    error << "Unable to find process group with name: ";
    error << name;
    error << " , instead we have ";
    error << pg_names_.size() << " process groups: {";
    for (const auto& pg : pg_names_) {
      error << static_cast<void*>(pg.first.get());
      error << " with name: ";
      error << pg.second;
      error << ", ";
    }
    error << "}";
    AT_ERROR(error.str());
  }

  TORCH_CHECK(it->first.defined(), "found a process group that's null");

  return it->first;
}

std::string DistributedC10d::getNameOfProcessGroup(const c10::intrusive_ptr<ProcessGroup>& pg) const {
  auto it = pg_names_.find(pg);
  if (it == pg_names_.end()) {
    std::stringstream error;
    error << "Unable to find name of process group ";
    error << static_cast<void*>(pg.get());
    error << "instead we have " << pg_names_.size() << " process groups: {";
    for (const auto& pg : pg_names_) {
      error << static_cast<void*>(pg.first.get());
      error << " with name: ";
      error << pg.second;
      error << ", ";
    }
    error << "}";
    AT_ERROR(error.str());
  }

  return it->second;
}

c10::intrusive_ptr<ProcessGroup> DistributedC10d::newProcessGroupHelper(
    const int64_t world_size,
    const int64_t rank,
    const std::vector<int64_t>& group_ranks,
    const std::string& backend_str,
    const c10::intrusive_ptr<Store>& store,
    c10::optional<std::string> group_name,
    int64_t timeout_milisesonds) {
  if (!group_name.has_value()) {
    group_name = std::to_string(group_count_);
    ++group_count_;
  }

  auto it = std::find_if(
      pg_names_.begin(),
      pg_names_.end(),
      [&](const std::pair<c10::intrusive_ptr<ProcessGroup>, std::string>&
              pg_name) { return pg_name.second == *group_name; });

  if (it != pg_names_.end()) {
    TORCH_CHECK(false,
        "The specified group name has already been "
        "created, please use a different group name");
  }

  bool is_default_group = (group_ranks.size() == 0);

  c10::intrusive_ptr<ProcessGroup> pg;

  auto timeout = std::chrono::milliseconds(timeout_milisesonds);

  std::string backend = Backend::get(backend_str);
  if (backend == "mpi") {
#ifdef USE_C10D_MPI
    std::vector<int> group_ranks_copy(group_ranks.begin(), group_ranks.end());
    pg = ProcessGroupMPI::createProcessGroupMPI(group_ranks_copy);
#else
    AT_ERROR(
        "Distributed package doesn't have MPI built in."
        " MPI is only included if you build PyTorch from"
        " source on a host that has MPI installed.");
#endif
  } else {
    if (!is_default_group) {
      int64_t global_rank = default_pg_->getRank();
      if (std::find(group_ranks.begin(), group_ranks.end(), global_rank) ==
          group_ranks.end()) {
        return pg;
      }
    }

    auto prefix_store = c10::make_intrusive<PrefixStore>(*group_name, store);

    if (backend == "gloo") {
#ifdef USE_C10D_GLOO
      auto options = ProcessGroupGloo::Options::create();

      // Use interfaces listed in "GLOO_SOCKET_IFNAME", if set.
      char* ifnameEnv = getenv(GLOO_SOCKET_IFNAME_ENV.c_str());
      if (ifnameEnv) {
        for (const auto& iface : split(',', ifnameEnv)) {
          options->devices.push_back(
              ::c10d::ProcessGroupGloo::createDeviceForInterface(iface));
        }
      } else {
        // If no hostname is specified, this function looks up
        // the machine's hostname and returns a device instance
        // associated with the address that the hostname resolves to.
        options->devices.push_back(
            ::c10d::ProcessGroupGloo::createDefaultDevice());
      }

      options->timeout = timeout;
      options->threads = options->devices.size() * 2;
      pg = c10::make_intrusive<ProcessGroupGloo>(
          prefix_store, rank, world_size, options);
#else
      AT_ERROR(
          "Attempting to create GLOO-based process group while GLOO is either not enabled or built");
#endif // USE_C10D_GLOO
    } else if (backend == "nccl") {
#ifdef USE_C10D_NCCL
      auto options = ProcessGroupNCCL::Options::create();

      options->is_high_priority_stream = false;
      options->timeout = timeout;
      pg = c10::make_intrusive<ProcessGroupNCCL>(
          prefix_store, rank, world_size, options);
#else
      AT_ERROR(
          "Attempting to create NCCL-based process group while NCCL is either not enabled or built");
#endif // USE_C10D_NCCL
    } else {
      // TODO: discuss to figure out how to extend this to third party backends?
      AT_ERROR("Unsupported backend type: ", backend);
    }
  }

  // register to process group map
  pg_map_[pg] = std::make_pair(backend, store);
  pg_names_[pg] = *group_name;
  return pg;
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

  AT_ERROR("The group rank is not part of the group");
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

void DistributedC10d::registerProcessGroupName(const c10::intrusive_ptr<ProcessGroup>& process_group, const std::string& name) {
  auto it = std::find_if(
      pg_names_.begin(),
      pg_names_.end(),
      [&](const std::pair<c10::intrusive_ptr<ProcessGroup>, std::string>&
              pg_name) { return pg_name.second == name; });

  if (it != pg_names_.end()) {
    TORCH_CHECK(
        it->first == process_group,
        "Requested name already exists: ",
        name,
        " and it is associated with a different process group");
    return;
  }

  it = pg_names_.find(process_group);
  TORCH_CHECK(
      it == pg_names_.end(),
      "Given process group has been registered before with a different name: ",
      it->second);

  pg_names_[process_group] = name;
}

void initCustomClassBindings() {
  static const auto StoreTorchBind =
      torch::class_<::c10d::Store>("dist_c10d", "Store");

  static const auto FileStoreTorchBind =
      torch::class_<::c10d::FileStore>("dist_c10d", "FileStore")
          .def(torch::init([](const std::string& path, int64_t num_workers) {
            return c10::make_intrusive<::c10d::FileStore>(path, num_workers);
          }));

  static const auto TCPStoreTorchBind =
      torch::class_<::c10d::TCPStore>("dist_c10d", "TCPStore")
          .def(torch::init([](const std::string& host_name,
                              int64_t port,
                              int64_t world_size,
                              bool is_master,
                              int64_t timeout) {
            auto timeout_miliseconds = std::chrono::milliseconds(timeout);
            return c10::make_intrusive<::c10d::TCPStore>(
                host_name, port, world_size, is_master, timeout_miliseconds);
          }));

  // TODO: This should really take Store as constructor argument instead of
  // TCPStore, but the fact that TorchScript does not support polymorphism
  // forced us to cast in C++ instead of automatic casting
  static const auto PrefixStoreTorchBind =
      torch::class_<::c10d::PrefixStore>("dist_c10d", "PrefixStore")
          .def(torch::init([](const std::string& prefix,
                              const c10::intrusive_ptr<::c10d::Store>& store) {
            return c10::make_intrusive<::c10d::PrefixStore>(prefix, store);
          }));

  // Torchbind the ProcessGroup to make it available in TorchScript
  static const auto ProcessGroupWorkTorchBind =
      torch::class_<::c10d::ProcessGroup::Work>("dist_c10d", "Work")
          .def(torch::init<>())
          .def(
              "wait",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup::Work>& work)
                  -> bool {
                // TODO: make std::chrono::millisecond works with TorchBind to
                // provide the full API in python
                return work->wait();
              })
          .def("result", &::c10d::ProcessGroup::Work::result);

  // TODO: Support argument names in Python API.
  static const auto ProcessGroupTorchBind =
      torch::class_<::c10d::ProcessGroup>("dist_c10d", "ProcessGroup")
          .def_pickle(
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self) {
                auto name =
                    ::c10d::DistributedC10d::get()->getNameOfProcessGroup(self);
                return std::vector<std::string>{name};
              },
              [](std::vector<std::string> state) {
                TORCH_CHECK(
                    state.size() == 1,
                    "Expecting exactly 1 state when restoring ProcessGroup, got: ",
                    state.size());
                const auto& process_group_name = state.front();
                auto process_group =
                    ::c10d::DistributedC10d::get()->getProcessGroupByName(
                        process_group_name);
                TORCH_CHECK(
                    process_group.defined(),
                    "Needed process group not found, ",
                    "please create a process group with name: ",
                    process_group_name);
                return process_group;
              })
          .def(
              "rank",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self) {
                return static_cast<int64_t>(self->getRank());
              })
          .def(
              "size",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self) {
                return static_cast<int64_t>(self->getSize());
              })
          // TODO: make BroadcastOptions compatible with TorchBind to provide
          // the full API in python.
          /*
          // TODO: Enable this method when TorchBind supports overloading.
          .def(
              "broadcast",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor> data) { return self->broadcast(data);
          })
          */
          .def(
              "broadcast",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor tensor,
                 int64_t rootRank) {
                ::c10d::BroadcastOptions opts;
                opts.rootRank = rootRank;
                std::vector<at::Tensor> tensors = {std::move(tensor)};
                return self->broadcast(tensors, opts);
              })
          // TODO: make AllreduceOptions compatible with TorchBind to provide
          // the full API in python.
          .def(
              "allreduce",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor> tensors) {
                return self->allreduce(tensors);
              })
          /*
          // TODO: Enable these methods when TorchBind supports overloading.
          // TODO: Enable these methods when ReduceOp can be torchbinded.
          .def(
              "allreduce",
              [](c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                  std::vector<at::Tensor>& tensors,
                  c10::intrusive_ptr<::c10d::ReduceOp> op) {
                      ::c10d::AllreduceOptions opts;
                      opts.reduceOp = *op;
                      return self->allreduce(tensors, opts);
                  }
          )
          .def(
              "allreduce",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor& tensor,
                 c10::intrusive_ptr<::c10d::ReduceOp> op) {
                      ::c10d::AllreduceOptions opts;
                      opts.reduceOp = *op;
                      std::vector<at::Tensor> tensors = {tensor};
                      return self->allreduce(tensors, opts);
                 }
           )
          */
          // TODO: make AllreduceCoalescedOptions compatible with TorchBind to
          // provide the full API in python.
          .def(
              "allreduce_coalesced",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor> tensors) {
                ::c10d::AllreduceCoalescedOptions opts;
                return self->allreduce_coalesced(tensors, opts);
              })
          .def(
              "reduce",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor> tensors) {
                ::c10d::ReduceOptions opts;
                return self->reduce(tensors, opts);
              })
          /*
          // TODO: Enable this when c10d::ReduceOp is TorchBind compatible.
          .def(
              "reduce",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
              at::Tensor& tensor,
              int rootRank,
              c10::intrusive_ptr<::c10d::ReduceOp> op) {
              ::c10d::ReduceOptions opts;
              opts.reduceOp = *op;
              opts.rootRank = rootRank;
              std::vector<at::Tensor> tensors = {tensor};
              return self->reduce(tensors, opts);
              })
          */
          .def(
              "allgather",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<std::vector<at::Tensor>> outputTensors,
                 std::vector<at::Tensor> inputTensors) {
                ::c10d::AllgatherOptions opts;
                return self->allgather(outputTensors, inputTensors, opts);
              })
          /*
          // TODO: Enable these methods when TorchBind supports overloading.
          .def(
              "allgather",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor> output,
                 at::Tensor input) {
                std::vector<std::vector<at::Tensor>> outputs = {
                    std::move(output)};
                std::vector<at::Tensor> inputs = {std::move(input)};
                ::c10d::AllgatherOptions opts;
                return self->allgather(outputs, inputs, opts);
              })
          */
          .def(
              "allgather_coalesced",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<std::vector<at::Tensor>> output_lists,
                 std::vector<at::Tensor> input_list) {
                ::c10d::AllgatherOptions opts;
                return self->allgather_coalesced(
                    output_lists, input_list, opts);
              })
          /*
          // TODO: Enable this method when TorchBind supports overloading.
          .def(
              "gather",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<std::vector<at::Tensor>> output_tensors,
                 std::vector<at::Tensor> input_tensors) {
                ::c10d::GatherOptions opts;
                return self->gather(output_tensors, input_tensors, opts);
              })
          */
          .def(
              "gather",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor> output,
                 at::Tensor input,
                 int64_t rootRank) {
                ::c10d::GatherOptions opts;
                opts.rootRank = rootRank;
                std::vector<std::vector<at::Tensor>> outputs = {
                    std::move(output)};
                std::vector<at::Tensor> inputs = {std::move(input)};
                return self->gather(outputs, inputs, opts);
              })
          /*
          // TODO: Enable this method when TorchBind supports overloading.
          .def(
              "scatter",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor> outputTensors,
                 std::vector<std::vector<at::Tensor>> inputTensors) {
                ::c10d::ScatterOptions opts;
                self->scatter(outputTensors, inputTensors, opts);
              })
          */
          .def(
              "scatter",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor output,
                 std::vector<at::Tensor> input,
                 int64_t rootRank) {
                ::c10d::ScatterOptions opts;
                opts.rootRank = rootRank;
                std::vector<std::vector<at::Tensor>> inputs = {
                    std::move(input)};
                std::vector<at::Tensor> outputs = {std::move(output)};
                return self->scatter(outputs, inputs, opts);
              })
          /*
          // TODO: Enable this method when TorchBind supports overloading.
          // TODO: Enable this method when TorchBind supports
          ReduceScatterOptions. .def( "reduce_scatter",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor> outputTensors,
                 std::vector<std::vector<at::Tensor>> inputTensors) {
                ::c10d::ReduceScatterOptions opts;
                return self->reduce_scatter(outputTensors, inputTensors, opts);
              })
          */
          .def(
              "reduce_scatter",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor output,
                 std::vector<at::Tensor> input) {
                std::vector<at::Tensor> outputs = {std::move(output)};
                std::vector<std::vector<at::Tensor>> inputs = {
                    std::move(input)};
                ::c10d::ReduceScatterOptions opts;
                return self->reduce_scatter(outputs, inputs, opts);
              })
          .def(
              "alltoall_base",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor outputTensor,
                 at::Tensor inputTensor,
                 std::vector<int64_t> outputSplitSizes,
                 std::vector<int64_t> inputSplitSizes) {
                ::c10d::AllToAllOptions opts;
                return self->alltoall_base(
                    outputTensor,
                    inputTensor,
                    outputSplitSizes,
                    inputSplitSizes,
                    opts);
              })
          .def(
              "alltoall",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor> outputTensors,
                 std::vector<at::Tensor> inputTensors) {
                ::c10d::AllToAllOptions opts;
                return self->alltoall(outputTensors, inputTensors, opts);
              })
          .def(
              "send",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor> tensors,
                 int64_t dstRank,
                 int64_t tag) {
                return self->send(
                    tensors, static_cast<int>(dstRank), static_cast<int>(tag));
              })
          .def(
              "recv",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor> tensors,
                 int64_t srcRank,
                 int64_t tag) {
                return self->recv(
                    tensors, static_cast<int>(srcRank), static_cast<int>(tag));
              })
          .def(
              "recv_anysource",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor> tensors,
                 int64_t tag) {
                return self->recvAnysource(tensors, static_cast<int>(tag));
              })
          .def(
              "barrier",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self) {
                ::c10d::BarrierOptions opts;
                return self->barrier(opts);
              });

#ifdef USE_C10D_NCCL
  // XXX: Ideally the Options of ProcessGroupNCCL should be
  // bound using `def_readwrite` like in pybind11, but we
  // didn't do that because: 1. no milisecond support yet
  // 2. no def_readwrite or property support yet.
  // TODO: make this binding the same as pybind11
  static const auto ProcessGroupNCCLOptionsTorchBind =
      torch::class_<::c10d::ProcessGroupNCCL::Options>(
          "dist_c10d", "ProcessGroupNCCLOptions")
          .def(torch::init([](int64_t timeout, bool isHighPriorityStream) {
            auto opTimeout = std::chrono::milliseconds(timeout);
            auto opts =
                ::c10d::ProcessGroupNCCL::Options::create(isHighPriorityStream);
            opts->timeout = opTimeout;
            return opts;
          }));

  static const auto ProcessGroupNCCLTorchBind =
      torch::class_<::c10d::ProcessGroupNCCL>("dist_c10d", "ProcessGroupNCCL")
          .def_pickle(
              [](const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& self) {
                auto base_process_group =
                    ::c10::static_intrusive_pointer_cast<::c10d::ProcessGroup>(self);
                auto name =
                    ::c10d::DistributedC10d::get()->getNameOfProcessGroup(self);
                return std::vector<std::string>{name};
              },
              [](std::vector<std::string> state) {
                TORCH_CHECK(
                    state.size() == 1,
                    "Expecting exactly 1 state when restoring ProcessGroupNCCL, got: ",
                    state.size());
                const auto& process_group_name = state.front();
                auto base_process_group =
                    ::c10d::DistributedC10d::get()->getProcessGroupByName(
                        process_group_name);
                TORCH_CHECK(
                    base_process_group.defined(),
                    "Needed process group not found, ",
                    "please create a process group with name: ",
                    process_group_name);
                c10::intrusive_ptr<::c10d::ProcessGroupNCCL>
                    process_group_nccl = ::c10::dynamic_intrusive_pointer_cast<
                        ::c10d::ProcessGroupNCCL>(base_process_group);
                TORCH_CHECK(
                    process_group_nccl.defined(),
                    "Process group ",
                    process_group_name,
                    " isn't configured for NCCL backend");
                return process_group_nccl;
              })
          .def(torch::init(
              [](const c10::intrusive_ptr<::c10d::Store>& store,
                 int64_t rank,
                 int64_t size,
                 c10::intrusive_ptr<::c10d::ProcessGroupNCCL::Options> options,
                 const std::string& name) {
                auto pg = c10::make_intrusive<::c10d::ProcessGroupNCCL>(
                    store, rank, size, options);
                ::c10d::DistributedC10d::get()->registerProcessGroupName(
                    pg, name);
                return pg;
              }))
          .def(
              "alltoall_base",
              [](const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& self,
                 at::Tensor output,
                 at::Tensor input,
                 std::vector<int64_t> outputSplitSizes,
                 std::vector<int64_t> inputSplitSizes) {
                return self->alltoall_base(
                    output,
                    input,
                    outputSplitSizes,
                    inputSplitSizes,
                    ::c10d::AllToAllOptions());
              })
          .def(
              "size",
              [](const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& self) {
                return (int64_t)self->getSize();
              })
          .def(
              "rank",
              [](const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& self) {
                return (int64_t)self->getRank();
              });
#endif

  static const auto DistributedC10dFrontendTorchBind =
      torch::class_<::c10d::DistributedC10d>("dist_c10d", "frontend")
          .def(torch::init([]() { return ::c10d::DistributedC10d::get(); }))
          .def(
              "new_process_group_helper",
              &::c10d::DistributedC10d::newProcessGroupHelper)
          .def(
              "get_process_group_by_name",
              &::c10d::DistributedC10d::getProcessGroupByName)
          .def(
              "get_name_of_process_group",
              &::c10d::DistributedC10d::getNameOfProcessGroup);
}

} // namespace c10d
