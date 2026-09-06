#pragma once

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <optional>

namespace c10d::nccl_ep {

// Dispatch output layout. Values mirror ncclEpLayout_t (ep_enums.h); kept as a
// standalone enum so this header stays free of <nccl_ep.h> (the .cu maps it to
// the library enum). Unset (0) is the zero-init sentinel.
enum class NcclEpLayout : int64_t {
  Unset = 0,
  ExpertMajor = 1,
  RankMajor = 2,
  Flat = 3,
};

struct NcclEpGroup : c10::intrusive_ptr_target {
  void* group{nullptr}; // ncclEpGroup_t, opaque to avoid including nccl_ep.h
  std::string group_name;

  NcclEpGroup() = default;
  ~NcclEpGroup();
};

struct NcclEpHandle : c10::intrusive_ptr_target {
  void* handle{nullptr}; // ncclEpHandle_t, opaque
  NcclEpLayout layout{NcclEpLayout::Unset}; // queried for output shapes
  std::string group_name; // for symm_mem zero-copy rendezvous lookup
  // The library stashes topk_idx's device pointer on the handle (per nccl_ep.h:
  // "User-owned (do not free). LL reads directly; HT uses cached
  // hybridep.topk_idx"). recv_total_counter is allocated by us and read back
  // by nccl_ep_handle_get_num_recv_tokens. Keep both alive for the handle's
  // lifetime so nccl_ep can't read freed memory.
  at::Tensor topk_idx;
  at::Tensor recv_total_counter;

  NcclEpHandle(
      void* handle,
      NcclEpLayout layout,
      std::string group_name,
      at::Tensor topk_idx,
      at::Tensor recv_total_counter)
      : handle(handle),
        layout(layout),
        group_name(std::move(group_name)),
        topk_idx(std::move(topk_idx)),
        recv_total_counter(std::move(recv_total_counter)) {}
  ~NcclEpHandle();
};

TORCH_API c10::intrusive_ptr<NcclEpGroup> nccl_ep_create_group(
    const c10::intrusive_ptr<::c10d::ProcessGroup>& pg,
    int64_t num_experts,
    int64_t max_dispatch_tokens_per_rank,
    int64_t max_recv_tokens_per_rank,
    int64_t max_token_bytes);

TORCH_API c10::intrusive_ptr<NcclEpHandle> nccl_ep_create_handle(
    const c10::intrusive_ptr<NcclEpGroup>& group,
    const at::Tensor& topk_idx,
    const std::optional<at::Tensor>& recv_expert_counter,
    NcclEpLayout layout);

TORCH_API int64_t nccl_ep_handle_get_num_recv_tokens(
    const c10::intrusive_ptr<NcclEpHandle>& handle);

// out_topk_idx is nullopt for expert-major layouts, which leave it
// unpopulated; out_topk_weights stays populated (1-D for expert-major, 2-D for
// flat) and is optional only for API symmetry.
TORCH_API void nccl_ep_dispatch(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& tokens,
    const at::Tensor& topk_weights,
    at::Tensor& out_tokens,
    std::optional<at::Tensor> out_topk_weights,
    std::optional<at::Tensor> out_topk_idx);

TORCH_API void nccl_ep_combine(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& expert_tokens,
    at::Tensor& out_tokens);

} // namespace c10d::nccl_ep
