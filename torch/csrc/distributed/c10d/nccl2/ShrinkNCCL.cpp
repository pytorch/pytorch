// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

#include <algorithm>
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
  options->enable_reconfigure = source->enable_reconfigure;
  return options;
}

} // namespace

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
    auto status = workq_.garbageCollect();
    TORCH_CHECK(
        status == WorkNCCL::WorkStatus::COMPLETED,
        "NCCL communicator cannot shrink while work is pending");
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
  auto shrinkStatus = nccl_api_->commShrink(
      nccl_comm_,
      excluded.data(),
      static_cast<int>(excluded.size()),
      &childComm,
      &childOptions->config,
      shrink_flags);
  try {
    waitForNcclChildComm(
        *nccl_api_,
        nccl_comm_,
        &childComm,
        shrinkStatus,
        true,
        childOptions->timeout,
        "NCCL commShrink failed");
  } catch (...) {
    comm_state_ = CommState::ERROR;
    nccl_comm_ = nullptr;
    throw;
  }

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

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
