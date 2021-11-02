#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <c10d/PrefixStore.hpp>
#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>

namespace c10d {

#ifdef USE_C10D_GLOO
static const std::string GLOO_SOCKET_IFNAME_ENV = "GLOO_SOCKET_IFNAME";
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
  static std::string get(const std::string&);

  // TODO: How to support registering third_party backend?
  static void registerBackend();

 private:
  // TODO: Should this be an enum list instead since this set doesn't
  // change at all.
  std::unordered_set<std::string> registered_backends_;
};

class TORCH_PYTHON_API DistributedC10d : public torch::CustomClassHolder {
 public:
  static c10::intrusive_ptr<DistributedC10d> get();

  DistributedC10d() = default;

  void initProcessGroup(
      const std::string& backend,
      const std::string& init_method,
      const std::chrono::milliseconds& timeout,
      int64_t world_size,
      int64_t rank,
      c10::intrusive_ptr<Store> store,
      const std::string& group_name);

  void destroyProcessGroup(c10::intrusive_ptr<ProcessGroup> group);
  int64_t getRank(const c10::intrusive_ptr<ProcessGroup>& group) const;
  int64_t getWorldSize(const c10::intrusive_ptr<ProcessGroup>& group) const;

  c10::intrusive_ptr<ProcessGroup::Work> isend(
      at::Tensor tensor,
      int64_t dst,
      const c10::intrusive_ptr<ProcessGroup>& group,
      c10::optional<int64_t>& tag);

  c10::intrusive_ptr<ProcessGroup::Work> irecv(
      at::Tensor tensor,
      int64_t src,
      const c10::intrusive_ptr<ProcessGroup>& group,
      c10::optional<int64_t>& tag);

  void send(
      at::Tensor tensor,
      int64_t dst,
      const c10::intrusive_ptr<ProcessGroup>& group,
      c10::optional<int64_t>& tag);

  int64_t recv(
      at::Tensor tensor,
      const c10::optional<int64_t>& src,
      const c10::intrusive_ptr<ProcessGroup>& group,
      c10::optional<int64_t>& tag);

  c10::intrusive_ptr<ProcessGroup::Work> broadcastMultiGPU(
      std::vector<at::Tensor>& tensor_list,
      int64_t src,
      const c10::intrusive_ptr<ProcessGroup>& group,
      bool async_op = false,
      int64_t src_tensor = 0);

  c10::intrusive_ptr<ProcessGroup::Work> broadcast(
      at::Tensor tensor,
      int64_t src,
      const c10::intrusive_ptr<ProcessGroup>& group,
      bool async_op = false);

  c10::intrusive_ptr<ProcessGroup::Work> allReduceMultiGPU(
      std::vector<at::Tensor>& tensor_list,
      const c10::intrusive_ptr<ProcessGroup>& group,
      ReduceOp op = ReduceOp::SUM,
      bool async_op = false);

  c10::intrusive_ptr<ProcessGroup::Work> allReduce(
      at::Tensor tensor,
      const c10::intrusive_ptr<ProcessGroup>& group,
      ReduceOp op = ReduceOp::SUM,
      bool async_op = false);

  c10::intrusive_ptr<ProcessGroup::Work> allReduceCoalesced(
      std::vector<at::Tensor>& tensors,
      const c10::intrusive_ptr<ProcessGroup>& group,
      ReduceOp op = ReduceOp::SUM,
      bool async_op = false);

  c10::intrusive_ptr<ProcessGroup::Work> reduceMultiGPU(
      std::vector<at::Tensor>& tensor_list,
      int64_t dst,
      const c10::intrusive_ptr<ProcessGroup>& group,
      ReduceOp op = ReduceOp::SUM,
      bool async_op = false,
      int64_t dst_tensor = 0);

  c10::intrusive_ptr<ProcessGroup::Work> reduce(
      at::Tensor tensor,
      int64_t dst,
      const c10::intrusive_ptr<ProcessGroup>& group,
      ReduceOp op = ReduceOp::SUM,
      bool async_op = false);

  c10::intrusive_ptr<ProcessGroup::Work> allGatherMultiGPU(
      std::vector<std::vector<at::Tensor>>& output_tensor_lists,
      std::vector<at::Tensor>& input_tensor_list,
      const c10::intrusive_ptr<ProcessGroup>& group,
      bool async_op = false);

  c10::intrusive_ptr<ProcessGroup::Work> allGather(
      std::vector<at::Tensor>& tensor_list,
      at::Tensor tensor,
      const c10::intrusive_ptr<ProcessGroup>& group,
      bool async_op = false);

  c10::intrusive_ptr<ProcessGroup::Work> allGatherCoalesced(
      std::vector<std::vector<at::Tensor>>& output_tensor_lists,
      std::vector<at::Tensor>& input_tensor_list,
      const c10::intrusive_ptr<ProcessGroup>& group,
      bool async_op = false);

  c10::intrusive_ptr<ProcessGroup::Work> gather(
      at::Tensor tensor,
      const c10::optional<std::vector<at::Tensor>>& gather_list,
      const c10::intrusive_ptr<ProcessGroup>& group,
      int64_t dst = 0,
      bool async_op = false);

  c10::intrusive_ptr<ProcessGroup::Work> scatter(
      at::Tensor tensor,
      std::vector<at::Tensor>& scatter_list,
      const c10::intrusive_ptr<ProcessGroup>& group,
      int64_t src = 0,
      bool async_op = false);

  c10::intrusive_ptr<ProcessGroup::Work> reduceScatterMultiGPU(
      std::vector<at::Tensor>& output_tensor_list,
      std::vector<std::vector<at::Tensor>>& input_tensor_lists,
      const c10::intrusive_ptr<ProcessGroup>& group,
      ReduceOp op = ReduceOp::SUM,
      bool async_op = false);

  c10::intrusive_ptr<ProcessGroup::Work> reduceScatter(
      at::Tensor output,
      std::vector<at::Tensor>& input_tensor_list,
      const c10::intrusive_ptr<ProcessGroup>& group,
      ReduceOp op = ReduceOp::SUM,
      bool async_op = false);

  c10::intrusive_ptr<ProcessGroup::Work> allToAllSingle(
      at::Tensor output,
      at::Tensor input,
      std::vector<int64_t>& output_split_sizes,
      std::vector<int64_t>& input_split_sizes,
      const c10::intrusive_ptr<ProcessGroup>& group,
      bool async_op = false);

  c10::intrusive_ptr<ProcessGroup::Work> allToAll(
      std::vector<at::Tensor>& output_tensor_list,
      std::vector<at::Tensor>& input_tensor_list,
      const c10::intrusive_ptr<ProcessGroup>& group,
      bool async_op = false);

  c10::intrusive_ptr<ProcessGroup::Work> barrier(
      const c10::intrusive_ptr<ProcessGroup>& group,
      bool async_op = false);

  c10::intrusive_ptr<ProcessGroup> newGroup(
      std::vector<int64_t> ranks,
      std::chrono::milliseconds timeout,
      Backend backend);

  c10::intrusive_ptr<ProcessGroup> worldProcessGroup();

  c10::intrusive_ptr<ProcessGroup> newProcessGroupHelper(
    const int64_t world_size,
    const int64_t rank,
    const std::vector<int64_t>& group_ranks,
    const std::string& backend_str,
    const c10::intrusive_ptr<Store>& store,
    c10::optional<std::string> group_name,
    int64_t timeout_milisesonds);

  c10::intrusive_ptr<ProcessGroup> getProcessGroupByName(
      const std::string& name) const;

  std::string getNameOfProcessGroup(
      const c10::intrusive_ptr<ProcessGroup>& pg) const;

    void registerProcessGroupName(const c10::intrusive_ptr<ProcessGroup>& process_group, const std::string& name);

 private:

  bool rankNotInGroup(const c10::intrusive_ptr<ProcessGroup>& group) const;
  int64_t getGroupRank(
      const c10::intrusive_ptr<ProcessGroup>& group,
      const int64_t rank) const;
  int64_t getGlobalRank(
      const c10::intrusive_ptr<ProcessGroup>& group,
      const int64_t group_rank) const;
  void checkDefaultPg() const;
  int64_t getGroupSize(const c10::intrusive_ptr<ProcessGroup>& group) const;
  std::string getBackend(const c10::intrusive_ptr<ProcessGroup>& group);

  std::string backend_;
  // TODO: Ask Alex what kind of equality we need. It determine whether we
  // need to use ProcessGroup or ProcesGroup* as key.
  std::unordered_map<
      c10::intrusive_ptr<ProcessGroup>,
      std::pair<std::string, c10::intrusive_ptr<Store>>>
      pg_map_;

  // Note, this is different mapping relationship than original Python
  // implementation.
  std::unordered_map<c10::intrusive_ptr<ProcessGroup>, std::string> pg_names_;

  // Process group's global rank to local rank mapping
  std::unordered_map<
      c10::intrusive_ptr<ProcessGroup>,
      std::unordered_map<int64_t, int64_t>>
      pg_group_ranks_;

  c10::intrusive_ptr<ProcessGroup> default_pg_;

  // Default value should be "env://"
  std::string default_pg_init_method_;

  int64_t group_count_;
};

// This class exists as a way to allow us to split NCCL-specific code into a
// different file. frontend_cuda.cpp will, if USE_C10D_NCCL is defined,
// override this NCCLProcessGroupProvider with one that will actually do
// something.
struct TORCH_API NCCLProcessGroupProvider {
  virtual c10::intrusive_ptr<ProcessGroup> get(
      c10::intrusive_ptr<PrefixStore> /*prefix_store*/,
      int64_t /*rank*/,
      int64_t /*world_size*/,
      std::chrono::milliseconds /*timeout*/) const {
    AT_ERROR(
        "Attempting to create NCCL-based process group while NCCL is either not enabled or built");
  }

  virtual ~NCCLProcessGroupProvider() = default;
};

TORCH_API void registerNCCLProcessGroupProvider(
    NCCLProcessGroupProvider* provider);

TORCH_API void initCustomClassBindings();

} // namespace c10d
