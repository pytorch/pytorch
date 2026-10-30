// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCLLazy.hpp>

#include <algorithm>

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>

namespace c10d::nccl2 {

namespace {

c10::intrusive_ptr<ProcessGroupNCCL> makePrimary(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const c10::intrusive_ptr<ProcessGroupNCCL::Options>& options) {
  TORCH_CHECK(
      !options->enable_reconfigure,
      "nccl-lazy does not support enable_reconfigure");
  return c10::make_intrusive<ProcessGroupNCCL>(store, rank, size, options);
}

ProcessGroupNCCLLazy::PairFactory makePairFactory(
    c10::intrusive_ptr<::c10d::Store> store,
    int rank,
    int size,
    const c10::intrusive_ptr<ProcessGroupNCCL::Options>& options) {
  // Snapshot the options now: the user's Options may be reused for another
  // group.
  auto timeout = options->timeout;
  auto high_priority = options->is_high_priority_stream;
  auto config = cloneNcclConfig(options->config);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
  // Name each pair comm by its pair, not the group's comm_name.
  config.commName = nullptr;
#endif
  return [timeout,
          high_priority,
          config,
          desc = options->group_desc,
          ranks = options->global_ranks_in_group,
          rank,
          size,
          store = std::move(store)](
             int pair_rank, int peer, const std::string& pair_name) {
    auto pair_store =
        c10::make_intrusive<::c10d::PrefixStore>(pair_name, store);
    auto pair_options = ProcessGroupNCCL::Options::create();
    pair_options->timeout = timeout;
    pair_options->is_high_priority_stream = high_priority;
    pair_options->config = cloneNcclConfig(config);
    pair_options->group_name = pair_name;
    pair_options->group_desc = desc;
    // World ranks of the pair, indexed by pair rank (the lower group rank is
    // pair rank 0). An empty or short map means this group spans the world.
    auto globalRank = [&](int r) {
      return ranks.size() < static_cast<size_t>(size) ? static_cast<uint64_t>(r)
                                                      : ranks[r];
    };
    pair_options->global_ranks_in_group = {
        globalRank(std::min(rank, peer)), globalRank(std::max(rank, peer))};
    return c10::make_intrusive<ProcessGroupNCCL>(
        pair_store, pair_rank, /*size=*/2, pair_options);
  };
}

} // namespace

ProcessGroupNCCLLazy::ProcessGroupNCCLLazy(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const c10::intrusive_ptr<ProcessGroupNCCL::Options>& options)
    : LazyBackend(
          rank,
          size,
          makePrimary(
              store,
              rank,
              size,
              options ? options : ProcessGroupNCCL::Options::create()),
          makePairFactory(
              store,
              rank,
              size,
              options ? options : ProcessGroupNCCL::Options::create())) {}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
