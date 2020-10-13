#pragma once

#include <ATen/ATen.h>
#include <bits/stdint-intn.h>
#include <c10/util/Optional.h>
#include <torch/lib/c10d/ProcessGroup.hpp>
#include <torch/lib/c10d/Store.hpp>
#include <torch/lib/c10d/Types.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>

namespace c10d {

#ifdef USE_C10D_GLOO
constexpr char* GLOO_SOCKET_IFNAME_ENV = "GLOO_SOCKET_IFNAME";
#endif

inline std::vector<std::string> split(
    char separator,
    const std::string& string) {
  std::vector<std::string> pieces;
  std::stringstream ss(string);
  std::string item;
  while (std::getline(ss, item, separator)) {
    pieces.push_back(std::move(item));
  }
  return pieces;
}

class Backend {
 public:
  // Maps to Backend.__new__ in Python.
  static std::string get(std::string);

  // TODO: How to support registering third_party backend?
  static void registerBackend();

 private:
  // TODO: Should this be an enum list instead since this set doesn't
  // change at all.
  std::unordered_set<std::string> registered_backends_;
};

class DistributedC10d {
 public:
  void destroyProcessGroup(std::shared_ptr<ProcessGroup> group);
  int64_t getRank(std::shared_ptr<ProcessGroup> group);
  int64_t getWorldSize(std::shared_ptr<ProcessGroup> group);

  // getters/setters for the global states
  const std::string& backend() const;
  void backend(std::string const& backend_name);

  const std::unordered_map<std::shared_ptr<ProcessGroup>, std::vector<int64_t>>&
  pg_group_ranks();
  void pg_group_ranks(std::unordered_map<
                      std::shared_ptr<ProcessGroup>,
                      std::vector<int64_t>> const& new_ranks);

  const std::string& default_pg_init_method() const;
  void default_pg_init_method(std::string const& init_method);

  ProcessGroup::Work isend(
      at::Tensor tensor,
      int64_t dst,
      std::shared_ptr<ProcessGroup> group,
      c10::optional<int64_t> tag);

  ProcessGroup::Work irecv(
      at::Tensor tensor,
      int64_t src,
      std::shared_ptr<ProcessGroup> group,
      c10::optional<int64_t> tag);

  ProcessGroup::Work send(
      at::Tensor tensor,
      int64_t dst,
      std::shared_ptr<ProcessGroup> group,
      c10::optional<int64_t> tag);

  ProcessGroup::Work recv(
      at::Tensor tensor,
      int64_t src,
      std::shared_ptr<ProcessGroup> group,
      c10::optional<int64_t> tag);

  c10::optional<ProcessGroup::Work> broadcastMultiGPU(
      std::vector<at::Tensor> tensor_list,
      int64_t src,
      std::shared_ptr<ProcessGroup> group,
      bool async_op,
      int64_t src_tensor);

  c10::optional<ProcessGroup::Work> broadcast(
      at::Tensor tensor,
      int64_t src,
      std::shared_ptr<ProcessGroup> group,
      bool async_op);

  c10::optional<ProcessGroup::Work> allReduceMultiGPU(
      std::vector<at::Tensor>& tensor_list,
      ReduceOp op,
      std::shared_ptr<ProcessGroup> group,
      bool async_op);

  c10::optional<ProcessGroup::Work> allReduce(
      at::Tensor tensor,
      ReduceOp op,
      std::shared_ptr<ProcessGroup> group,
      bool async_op);

  c10::optional<ProcessGroup::Work> allReduceCoalesced(
      at::Tensor tensor,
      ReduceOp op,
      std::shared_ptr<ProcessGroup> group,
      bool async_op);

  c10::optional<ProcessGroup::Work> reduceMultiGPU(
      std::vector<at::Tensor>& tensor_list,
      int64_t dst,
      ReduceOp op,
      std::shared_ptr<ProcessGroup> group,
      bool async_op,
      int64_t dst_tensor);

  c10::optional<ProcessGroup::Work> reduce(
      at::Tensor tensor,
      int64_t dst,
      ReduceOp op,
      std::shared_ptr<ProcessGroup>& group,
      bool async_op);

  c10::optional<ProcessGroup::Work> allGatherMultiGPU(
      std::vector<std::vector<at::Tensor>>& output_tensor_lists,
      const std::vector<at::Tensor>& input_tensor_list,
      std::shared_ptr<ProcessGroup> group,
      bool async_op);

    // TODO TODO following APIs take python objects and unpickle them, how do we support these?
    // ProcessGroup::Work allGatherObject()
    // ProcessGroup::Work gatherObject()
    // ProcessGroup::Work broadcastObjectList()

  c10::optional<ProcessGroup::Work> allGather(
      std::vector<at::Tensor>& tensor_list,
      at::Tensor tensor,
      std::shared_ptr<ProcessGroup> group,
      bool async_op);

  c10::optional<ProcessGroup::Work> allGatherCoalesced(
      std::vector<std::vector<at::Tensor>>& output_tensor_lists,
      std::vector<at::Tensor>& input_tensor_list,
      std::shared_ptr<ProcessGroup> group,
      bool async_op);

  c10::optional<ProcessGroup::Work> gather(
      at::Tensor tensor,
      std::vector<at::Tensor>& gather_list,
      int64_t dst,
      std::shared_ptr<ProcessGroup> group,
      bool async_op);

  c10::optional<ProcessGroup::Work> scatter(
      at::Tensor tensor,
      std::vector<at::Tensor>& scatter_list,
      int64_t dst,
      std::shared_ptr<ProcessGroup> group,
      bool async_op);

  ProcessGroup::Work reduceScatterMultiGPU(
      std::vector<at::Tensor>& output_tensor_list,
      const std::vector<std::vector<at::Tensor>>& input_tensor_lists,
      ReduceOp op,
      std::shared_ptr<ProcessGroup> group,
      bool async_op);

  ProcessGroup::Work reduceScatter(
      at::Tensor output,
      const std::vector<at::Tensor>& input_list,
      ReduceOp op,
      std::shared_ptr<ProcessGroup> group,
      bool async_op);

  ProcessGroup::Work allToAllSingle(
      at::Tensor output,
      at::Tensor input,
      const std::vector<int64_t>& output_split_sizes,
      const std::vector<int64_t>& input_split_sizes,
      std::shared_ptr<ProcessGroup> group,
      bool async_op);

  ProcessGroup::Work allToAll(
      std::vector<at::Tensor>& output_tensor_list,
      const std::vector<at::Tensor>& input_tensor_list,
      std::shared_ptr<ProcessGroup> group,
      bool async_op);

  ProcessGroup::Work barrier(
      std::shared_ptr<ProcessGroup> group,
      bool async_op);

  std::shared_ptr<ProcessGroup> newGroup(
      std::vector<int64_t> ranks,
      std::chrono::milliseconds timeout,
      Backend backend);

 private:
  DistributedC10d(){};

  bool rankNotInGroup(std::shared_ptr<ProcessGroup> group) const;
  int64_t getGroupRank(std::shared_ptr<ProcessGroup> group, const int64_t rank)
      const;
  int64_t getGlobalRank(
      std::shared_ptr<ProcessGroup> group,
      const int64_t global_rank) const;
  void checkDefaultPg() const;
  int64_t getGroupSize(std::shared_ptr<ProcessGroup> group) const;
  int64_t getBackend(std::shared_ptr<ProcessGroup> group);

  std::shared_ptr<ProcessGroup> newProcessGroupHelper(
      const int64_t world_size,
      const int64_t rank,
      const std::vector<int64_t>& group_ranks,
      const std::string& backend_str,
      const std::shared_ptr<Store>& store,
      c10::optional<std::string> group_name,
      std::chrono::milliseconds timeout);

  std::string backend_;
  // TODO: Ask Alex what kind of equality we need. It determine whether we
  // need to use ProcessGroup or ProcesGroup* as key.
  std::unordered_map<
      std::shared_ptr<ProcessGroup>,
      std::pair<std::string, std::shared_ptr<Store>>>
      pg_map_;

  // Note, this is different mapping relationship than original Python
  // implementation.
  std::unordered_map<std::shared_ptr<ProcessGroup>, std::string> pg_names_;

  // Value is global_rank:group_rank mapping.
  std::unordered_map<std::shared_ptr<ProcessGroup>, std::vector<int64_t>>
      pg_group_ranks_;

  std::shared_ptr<ProcessGroup> default_pg_;

  // Default value should be "env://"
  std::string default_pg_init_method_;

  int64_t group_count_;
};

} // namespace c10d
