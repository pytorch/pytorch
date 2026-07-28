// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/lazy/LazyBackend.hpp>
#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

namespace c10d::nccl2 {

// "nccl-lazy" backend: ProcessGroupNCCL wrapped in the generic
// c10d::LazyBackend so each point-to-point peer pair gets its own
// lazily-created 2-rank ncclComm (and therefore its own stream). See
// lazy/LazyBackend.hpp for the wrapper semantics.
//
// A pair comm is just another ProcessGroupNCCL bootstrapped over a
// PrefixStore carved out of the caller's store, keyed by the wrapper's unique
// pair name (c10d PGs boot from a Store, so no bespoke ncclUniqueId exchange
// is needed, unlike torchcomms' createPairComm). The NCCL bootstrap of a pair
// comm stays lazy: it runs on the first send/recv, which knows the tensor's
// device.
class TORCH_API ProcessGroupNCCLLazy
    : public ::c10d::LazyBackend<ProcessGroupNCCL> {
 public:
  static constexpr std::string_view kBackendName = "nccl-lazy";

  ProcessGroupNCCLLazy(
      c10::intrusive_ptr<::c10d::Store> store,
      int rank,
      int size,
      c10::intrusive_ptr<ProcessGroupNCCL::Options> options =
          ProcessGroupNCCL::Options::create());

  const std::string getBackendName() const override {
    return std::string(kBackendName);
  }
};

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
