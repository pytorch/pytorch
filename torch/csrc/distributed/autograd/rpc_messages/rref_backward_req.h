#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>

namespace torch::distributed::autograd {

// Internal system RPC to invoke distributed backward pass on remote nodes when
// 'rref.backward()' is invoked.
class TORCH_API RRefBackwardReq : public rpc::RpcCommandBase {
 public:
  RRefBackwardReq(
      const rpc::RRefId& rrefId,
      int64_t autogradContextId,
      bool retainGraph = false);

  const rpc::RRefId& getRRefId() const;

  int64_t getAutogradContextId() const;

  bool retainGraph() const;

  // Serialization and deserialization methods.
  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;
  static std::unique_ptr<RRefBackwardReq> fromMessage(
      const rpc::Message& message);

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const rpc::RRefId rrefId_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const int64_t autogradContextId_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const bool retainGraph_;
};

} // namespace torch::distributed::autograd
