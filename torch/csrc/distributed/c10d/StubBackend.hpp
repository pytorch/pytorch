#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d {

// Minimal C++ Backend for testing. Modeled after ncclx: a custom backend
// plugin that registers via Backend.register_backend() with extended_api=True.
// All collectives are no-ops. Used to test code paths where a custom CUDA
// backend is present but the default ProcessGroup backend type may be
// UNDEFINED (e.g., multi-backend configs like "cpu:gloo,cuda:stub").
class StubWork : public Work {
 public:
  bool wait(std::chrono::milliseconds /* timeout */ = kNoTimeout) override {
    return true;
  }

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    auto fut = c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());
    fut->markCompleted();
    return fut;
  }
};

class StubBackend : public Backend {
 public:
  struct Options : Backend::Options {
    explicit Options() : Backend::Options("stub") {}

    std::vector<uint64_t> global_ranks_in_group;
    std::string group_name;
  };

  StubBackend(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = c10::make_intrusive<Options>())
      : Backend(rank, size), store_(store), options_(std::move(options)) {
    this->setGroupUid(options_->group_name);
  }

  const std::string getBackendName() const override {
    return "stub";
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>&,
      const BroadcastOptions& = BroadcastOptions()) override {
    return c10::make_intrusive<StubWork>();
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>&,
      const AllreduceOptions& = AllreduceOptions()) override {
    return c10::make_intrusive<StubWork>();
  }

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>&,
      std::vector<at::Tensor>&,
      const AllgatherOptions& = AllgatherOptions()) override {
    return c10::make_intrusive<StubWork>();
  }

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& = BarrierOptions()) override {
    return c10::make_intrusive<StubWork>();
  }

 private:
  c10::intrusive_ptr<Store> store_;
  c10::intrusive_ptr<Options> options_;
};

} // namespace c10d
