// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef USE_C10D_NCCL

#include <ATen/ATen.h>
#include <vector>

namespace c10d::nccl2 {

/**
 * BatchSendRecv accumulates point-to-point operations (sends and receives) so
 * they can be issued as a single grouped (ncclGroupStart/End) collective.
 *
 * In upstream torchcomms this held a shared_ptr<TorchComm> and issued through
 * it. With the c10d collapse there is no TorchComm: this is a plain op
 * container, and ProcessGroupNCCL owns issuing via batch_op_issue(ops, ...).
 */
class BatchSendRecv {
 public:
  class P2POp {
   public:
    enum class OpType { SEND, RECV };
    P2POp(OpType type, const at::Tensor& tensor, int peer)
        : type(type), tensor(tensor), peer(peer) {}

    OpType type;
    at::Tensor tensor;
    int peer;
  };

  void send(const at::Tensor& tensor, int dst) {
    ops.emplace_back(P2POp::OpType::SEND, tensor, dst);
  }
  void recv(const at::Tensor& tensor, int src) {
    ops.emplace_back(P2POp::OpType::RECV, tensor, src);
  }

  std::vector<P2POp> ops;
};

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
