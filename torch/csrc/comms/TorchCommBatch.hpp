// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/comms/TorchCommOptions.hpp>
#include <memory>

namespace torch::comms {

// Forward declaration
class TorchComm;
class TorchWork;

/**
 * BatchSendRecv enables batching multiple point-to-point operations
 * (sends and receives) into a single collective call.
 *
 * Lifetime Safety:
 * This class holds a shared_ptr to the parent TorchComm to ensure the
 * communicator remains valid for the lifetime of the batch. Users can
 * safely destroy the original TorchComm reference after creating a batch,
 * as long as the batch object itself remains alive until issue() completes.
 */
class BatchSendRecv {
 public:
  explicit BatchSendRecv(std::shared_ptr<TorchComm> parent);
  ~BatchSendRecv() = default;
  BatchSendRecv(const BatchSendRecv&) = default;
  BatchSendRecv& operator=(const BatchSendRecv&) = default;
  BatchSendRecv(BatchSendRecv&&) = default;
  BatchSendRecv& operator=(BatchSendRecv&&) = default;

  void send(const at::Tensor& tensor, int dst);
  void recv(at::Tensor& tensor, int src);
  c10::intrusive_ptr<TorchWork> issue(
      bool async_op,
      const BatchP2POptions& options = {});

  class P2POp {
   public:
    enum class OpType { SEND, RECV };
    P2POp(OpType type, const at::Tensor& tensor, int peer);
    ~P2POp() = default;
    P2POp(const P2POp&) = default;
    P2POp& operator=(const P2POp&) = default;
    P2POp(P2POp&&) = default;
    P2POp& operator=(P2POp&&) = default;

    OpType type;
    at::Tensor tensor;
    int peer;
  };

  std::vector<P2POp> ops;

 private:
  std::shared_ptr<TorchComm> parent_;
};

} // namespace torch::comms
