// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCLLazy.hpp>

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>

namespace c10d::nccl2 {

namespace {

c10::intrusive_ptr<ProcessGroupNCCL> makePrimary(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const c10::intrusive_ptr<ProcessGroupNCCL::Options>& options) {
  return c10::make_intrusive<ProcessGroupNCCL>(store, rank, size, options);
}

ProcessGroupNCCLLazy::PairFactory makePairFactory(
    c10::intrusive_ptr<::c10d::Store> store,
    c10::intrusive_ptr<ProcessGroupNCCL::Options> options) {
  return [store = std::move(store), options = std::move(options)](
             int pair_rank, const std::string& pair_name, at::Device device) {
    auto pair_store =
        c10::make_intrusive<::c10d::PrefixStore>(pair_name, store);
    auto pair_options = ProcessGroupNCCL::Options::create();
    pair_options->timeout = options->timeout;
    pair_options->is_high_priority_stream = options->is_high_priority_stream;
    pair_options->enable_reconfigure = options->enable_reconfigure;
    pair_options->config = cloneNcclConfig(options->config);
    pair_options->group_name = pair_name;
    auto pair = c10::make_intrusive<ProcessGroupNCCL>(
        pair_store, pair_rank, /*size=*/2, pair_options);
    if (pair_options->enable_reconfigure) {
      pair->setBoundDeviceId(device);
      pair_store->set(
          c10::str("reconfigure_handle_", pair_rank),
          pair->get_reconfigure_handle());
      std::vector<::c10d::ReconfigureHandle> handles;
      handles.reserve(2);
      for (int rank = 0; rank < 2; ++rank) {
        handles.push_back(
            pair_store->get_to_str(c10::str("reconfigure_handle_", rank)));
      }
      ::c10d::ReconfigureOptions reconfigure_options;
      reconfigure_options.handles = std::move(handles);
      reconfigure_options.timeout = pair_options->timeout;
      pair->reconfigure(reconfigure_options)->wait();
    }
    return pair;
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
              options ? options : ProcessGroupNCCL::Options::create())) {}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
