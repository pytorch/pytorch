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
             int pair_rank, const std::string& pair_name) {
    auto pair_store =
        c10::make_intrusive<::c10d::PrefixStore>(pair_name, store);
    auto pair_options = ProcessGroupNCCL::Options::create();
    pair_options->timeout = options->timeout;
    pair_options->is_high_priority_stream = options->is_high_priority_stream;
    pair_options->abort_process_on_timeout_or_error =
        options->abort_process_on_timeout_or_error;
    pair_options->hints = options->hints;
    pair_options->group_name = pair_name;
    return c10::make_intrusive<ProcessGroupNCCL>(
        pair_store, pair_rank, /*size=*/2, pair_options);
  };
}

} // namespace

ProcessGroupNCCLLazy::ProcessGroupNCCLLazy(
    c10::intrusive_ptr<::c10d::Store> store,
    int rank,
    int size,
    c10::intrusive_ptr<ProcessGroupNCCL::Options> options)
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
