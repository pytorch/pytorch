#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>

#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/debug.h>
#include <torch/csrc/distributed/c10d/sequence_num.hpp>

// *************************************************************************
// PROCESS GROUP collective communication API IS BEING CHANGED BETWEEN
// versions 1.7 and 1.8.
// PLEASE DO NOT ADD ANY DEPENDENCIES.
// SEE RFC: https://github.com/pytorch/pytorch/issues/39662
// *************************************************************************

constexpr auto kProcessGroupDefaultTimeout =
    std::chrono::milliseconds(30 * 60 * 1000);

namespace c10d {

// ProcessGroup is a base class that captures collective and point to
// point communication in a fixed set of processes.
//
// The functions specified in the class below describe the API alone;
// implementations are provided in subclasses.
//
// Every function that performs I/O is executed asynchronously by a
// thread pool owned by the ProcessGroup (by default). They return an
// object that can be used to wait for completion or error.
//
// The ProcessGroup can instantiate subgroups with fewer or an equal
// number of members. Implementations must take care that multiple
// process groups can be used in parallel and synchronize accordingly.
//
// The ProcessGroup assumes a fixed set of processes. If the set
// changes, existing instances must be destructed and instantiation
// and initialization must start from scratch. For members of the
// process group to find each other (referred to as rendezvous from
// hereon)
//
class TORCH_API ProcessGroup : public torch::CustomClassHolder {
 public:
  // ProcessGroup Options is a base struct that defines the basic options
  // when constructing a ProcessGroup. Each ProcessGroup subclass should
  // extend this struct and define its options if it wants to provide more
  // config options (beyond basic ones defined here) to end user.
  struct TORCH_API Options : torch::CustomClassHolder {
    explicit Options(
        std::string backend,
        std::chrono::milliseconds timeout = kProcessGroupDefaultTimeout)
        : timeout(timeout), backend(backend) {}
    virtual ~Options() = default;

    std::chrono::milliseconds timeout;

    // backend name
    const std::string backend;
  };

  explicit ProcessGroup(int rank, int size);
  virtual ~ProcessGroup();

  int getRank() const {
    return rank_;
  }

  int getSize() const {
    return size_;
  }

  // Subclasses must override this method to return the backend name
  virtual const std::string getBackendName() const {
    TORCH_INTERNAL_ASSERT(false, "getBackendName is not implemented.");
  };

  virtual void startCoalescing() {
    // no-op for backends that have not implemented startCoalescing
  }

  virtual void endCoalescing(
      std::vector<c10::intrusive_ptr<Work>>& /* reqs */) {
    // no-op for backends that have not implemented endCoalescing
  }

  // Consider using ops in Ops.hpp instead of the below, which route things
  // to the dispatcher.
  // TODO: Find a way to force the above rule programmatically.
  virtual c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& /* tensors */,
      const BroadcastOptions& /* opts */ = BroadcastOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ", getBackendName(), " does not support broadcast"));
  }

  virtual c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& /* tensors */,
      const AllreduceOptions& /* opts */ = AllreduceOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ", getBackendName(), " does not support allreduce"));
  }

  virtual c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& /* tensors */,
      const AllreduceCoalescedOptions& /* opts */ =
          AllreduceCoalescedOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ",
            getBackendName(),
            " does not support allreduce_coalesced"));
  }

  virtual c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& /* tensors */,
      const ReduceOptions& /* opts */ = ReduceOptions()) {
    TORCH_CHECK(
        false,
        c10::str("ProcessGroup ", getBackendName(), "does not support reduce"));
  }

  virtual c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ", getBackendName(), " does not support allgather"));
  }

  // Gathers a single tensor inputBuffer into a single buffer outputBuffer that
  // is interpreted as a contigious collection of size inputBuffer * WORLD_SIZE.
  // For implementers of ProcessGroup API and advanced users only.
  // Note: this function will be deprecated in near future.
  virtual c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& /* outputBuffer */,
      at::Tensor& /* inputBuffer */,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ",
            getBackendName(),
            " does not support _allgather_base"));
  }

  // This function is deprecated and will be moved out of ProcessGroup to comms:
  // * do not add dependencies on this function,
  // * do not implement it in your ProcessGroup, implement _allgather_base
  //   instead.
  virtual c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& /* outputTensorLists */,
      std::vector<at::Tensor>& /* inputTensors */,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ",
            getBackendName(),
            " does not support allgather_coalesced"));
  }

  virtual c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
      const GatherOptions& /* opts */ = GatherOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ", getBackendName(), " does not support gather"));
  }

  virtual c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<std::vector<at::Tensor>>& /* inputTensors */,
      const ScatterOptions& /* opts */ = ScatterOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ", getBackendName(), " does not support scatter"));
  }

  virtual c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<std::vector<at::Tensor>>& /* inputTensors */,
      const ReduceScatterOptions& /* opts */ = ReduceScatterOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ",
            getBackendName(),
            " does not support reduce_scatter"));
  }

  virtual c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& /* outputBuffer */,
      at::Tensor& /* inputBuffer */,
      const ReduceScatterOptions& /* opts */ = ReduceScatterOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ",
            getBackendName(),
            " does not support _reduce_scatter_base"));
  }

  virtual c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& /* outputBuffer */,
      at::Tensor& /* inputBuffer */,
      std::vector<int64_t>& /* outputSplitSizes */,
      std::vector<int64_t>& /* inputSplitSizes */,
      const AllToAllOptions& /* opts */ = AllToAllOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ",
            getBackendName(),
            " does not support alltoall_base"));
  }

  virtual c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
      const AllToAllOptions& opts = AllToAllOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ", getBackendName(), " does not support alltoall"));
  }

  virtual void monitoredBarrier(
      const BarrierOptions& /* unused */,
      bool /* unused */ = false) {
    auto backendName = getBackendName();
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ",
            backendName,
            " does not support monitoredBarrier, only GLOO supports monitored barrier."));
  }

  // Agrees on an initial sequence number for the whole group by having rank 0
  // create it and broadcast it to other ranks using the store. Only implemented
  // for GLOO and NCCL backends currently.
  virtual void setSequenceNumberForGroup() {
    auto backendName = getBackendName();
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ",
            backendName,
            " does not yet support sequence numbers."));
  }

  // Retrieves the current sequence number for the whole group, which should be
  // in sync. If the returned number is not consistent across the group, it
  // may indicate that there is some sort of collective desynchronization.
  virtual uint64_t getSequenceNumberForGroup() {
    auto backendName = getBackendName();
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ",
            backendName,
            " does not yet support sequence numbers."));
  }

  virtual c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& /* tensors */,
      int /* dstRank */,
      int /* tag */) {
    TORCH_CHECK(
        false,
        c10::str("ProcessGroup ", getBackendName(), " does not support send"));
  }

  virtual c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& /* tensors */,
      int /* srcRank */,
      int /* tag */) {
    TORCH_CHECK(
        false,
        c10::str("ProcessGroup ", getBackendName(), " does not support recv"));
  }

  virtual c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& /* tensors */,
      int /* tag */) {
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ",
            getBackendName(),
            " does not support recvAnysource"));
  }

  virtual c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& /* opts */ = BarrierOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "ProcessGroup ", getBackendName(), " does not support barrier"));
  }

 protected:
  // Implementations of this interface need to call this to setup
  // appropriate logging etc.
  void init();

  const int rank_;
  const int size_;
  // Optional sequence number structure for matching collectives.
  c10::optional<c10d::SequenceNum> sequenceNum_ = c10::nullopt;
  // Debug level setting. It is parsed once when ProcessGroup is constructed and
  // remains the same across use of this process group.
  DebugLevel dist_debug_level_;
};

} // namespace c10d
