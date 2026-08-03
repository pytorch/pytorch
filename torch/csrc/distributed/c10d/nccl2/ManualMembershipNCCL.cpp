// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

#include <algorithm>
#include <cstring>
#include <unordered_set>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace c10d::nccl2 {

namespace {

c10::intrusive_ptr<ProcessGroupNCCL::Options> cloneOptions(
    const c10::intrusive_ptr<ProcessGroupNCCL::Options>& source) {
  auto options =
      ProcessGroupNCCL::Options::create(source->is_high_priority_stream);
  options->timeout = source->timeout;
  options->config = cloneNcclConfig(source->config);
  options->group_name = source->group_name;
  options->group_desc = source->group_desc;
  options->global_ranks_in_group = source->global_ranks_in_group;
  options->use_pg_for_symm_mem_rendezvous =
      source->use_pg_for_symm_mem_rendezvous;
  options->enable_reconfigure = source->enable_reconfigure;
  return options;
}

} // namespace

void ProcessGroupNCCL::checkNoPendingWork() {
  auto status = workq_.garbageCollect();
  TORCH_CHECK(
      status == WorkNCCL::WorkStatus::COMPLETED,
      "NCCL communicator membership cannot change while work is pending");
}

c10::intrusive_ptr<::c10d::Backend> ProcessGroupNCCL::shrink(
    const std::vector<int64_t>& ranks_to_exclude,
    int shrink_flags,
    const c10::intrusive_ptr<::c10d::Backend::Options>& opts_override) {
  TORCH_CHECK(supportsShrinking(), "nccl2 shrink requires NCCL 2.27 or later");
  TORCH_CHECK_VALUE(
      !ranks_to_exclude.empty(), "ranks_to_exclude cannot be empty");
  TORCH_CHECK_VALUE(
      ranks_to_exclude.size() < static_cast<size_t>(getSize()),
      "Cannot exclude all ranks from an NCCL communicator");
  TORCH_CHECK_VALUE(
      shrink_flags == 0 || shrink_flags == NCCL_SHRINK_ABORT,
      "Invalid NCCL shrink flags: ",
      shrink_flags);

  if (init_state_ != InitializationState::INITIALIZED) {
    auto device = getBoundDeviceId().value_or(
        at::Device(at::kCUDA, at::cuda::current_device()));
    ensureInitialized(device);
  }
  checkAndAbortIfTimedOutOrError();

  std::unordered_set<int> seen;
  std::vector<int> excluded;
  excluded.reserve(ranks_to_exclude.size());
  for (auto rank : ranks_to_exclude) {
    TORCH_CHECK_VALUE(
        rank >= 0 && rank < getSize(), "Invalid rank ", rank, " for shrink");
    auto intRank = static_cast<int>(rank);
    TORCH_CHECK_VALUE(
        seen.insert(intRank).second, "Duplicate shrink rank ", rank);
    excluded.push_back(intRank);
  }
  TORCH_CHECK_VALUE(
      !seen.contains(getRank()),
      "An excluded rank must not call ProcessGroupNCCL::shrink");
  if (shrink_flags == 0) {
    checkNoPendingWork();
  }

  auto overrideOptions =
      c10::dynamic_intrusive_pointer_cast<Options>(opts_override);
  TORCH_CHECK(
      !opts_override || overrideOptions,
      "nccl2 shrink options must be ProcessGroupNCCL2.Options");
  auto childOptions =
      cloneOptions(overrideOptions ? overrideOptions : options_c10d_);

  c10::cuda::CUDAGuard guard(device_);
  ncclComm_t childComm = nullptr;
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commShrink(
          nccl_comm_,
          excluded.data(),
          static_cast<int>(excluded.size()),
          &childComm,
          &childOptions->config,
          shrink_flags),
      "NCCL commShrink failed");

  auto excludedBeforeRank = static_cast<int>(std::ranges::count_if(
      excluded, [this](int rank) { return rank < getRank(); }));
  auto child = c10::make_intrusive<ProcessGroupNCCL>(
      store_->clone(),
      getRank() - excludedBeforeRank,
      getSize() - static_cast<int>(excluded.size()),
      childOptions);
  child->initFromComm(childComm, device_, nccl_api_);
  return c10::static_intrusive_pointer_cast<::c10d::Backend>(child);
}

std::vector<uint8_t> ProcessGroupNCCL::getGrowId() {
  TORCH_CHECK(
      NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0),
      "nccl2 grow requires NCCL 2.29 or later");
  TORCH_CHECK(
      init_state_ == InitializationState::INITIALIZED,
      "Cannot get a grow ID from an uninitialized communicator");
  checkAndAbortIfTimedOutOrError();
  checkNoPendingWork();

  ncclUniqueId id{};
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commGetUniqueId(nccl_comm_, &id),
      "NCCL commGetUniqueId failed");
  return std::vector<uint8_t>(
      reinterpret_cast<uint8_t*>(&id),
      reinterpret_cast<uint8_t*>(&id) + sizeof(id));
}

c10::intrusive_ptr<ProcessGroupNCCL> ProcessGroupNCCL::grow(
    int new_size,
    const std::optional<std::vector<uint8_t>>& grow_id,
    int new_rank,
    const c10::intrusive_ptr<Options>& opts_override) {
  TORCH_CHECK(
      NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0),
      "nccl2 grow requires NCCL 2.29 or later");
  const bool existingRank = init_state_ == InitializationState::INITIALIZED;
  if (existingRank) {
    TORCH_CHECK(new_size > getSize(), "Grow size must exceed the current size");
    TORCH_CHECK(new_rank == -1, "Existing ranks must use new_rank=-1");
    TORCH_CHECK(!grow_id, "Existing ranks must not pass a grow ID");
    checkAndAbortIfTimedOutOrError();
    checkNoPendingWork();
  } else {
    TORCH_CHECK(
        init_state_ == InitializationState::UNINITIALIZED,
        "Cannot grow a finalized communicator");
    TORCH_CHECK(
        grow_id && grow_id->size() == sizeof(ncclUniqueId),
        "New ranks require a valid NCCL grow ID");
    TORCH_CHECK(
        new_rank >= 0 && new_rank < new_size,
        "Invalid new rank ",
        new_rank,
        " for grow size ",
        new_size);
    if (!nccl_api_) {
      nccl_api_ = std::make_shared<DefaultNcclApi>();
    }
  }

  auto childOptions =
      cloneOptions(opts_override ? opts_override : options_c10d_);
  auto device = existingRank ? device_
                             : getBoundDeviceId().value_or(at::Device(
                                   at::kCUDA, at::cuda::current_device()));
  c10::cuda::CUDAGuard guard(device);

  ncclUniqueId id{};
  const ncclUniqueId* idPtr = nullptr;
  if (grow_id) {
    std::memcpy(&id, grow_id->data(), sizeof(id));
    idPtr = &id;
  }

  ncclComm_t childComm = nullptr;
  NCCL_CHECK(
      nccl_api_,
      existingRank ? nccl_comm_ : nullptr,
      nccl_api_->commGrow(
          existingRank ? nccl_comm_ : nullptr,
          new_size,
          idPtr,
          new_rank,
          &childComm,
          &childOptions->config),
      "NCCL commGrow failed");

  auto child = c10::make_intrusive<ProcessGroupNCCL>(
      store_->clone(),
      existingRank ? getRank() : new_rank,
      new_size,
      childOptions);
  child->initFromComm(childComm, device, nccl_api_);
  return child;
}

void ProcessGroupNCCL::revoke() {
  TORCH_CHECK(
      init_state_ == InitializationState::INITIALIZED,
      "Cannot revoke an uninitialized communicator");
  revokeNcclComm();
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
