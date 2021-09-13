#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/types.h>

namespace torch {
namespace distributed {
namespace rpc {

// Base class for all RPC request and responses.
class RpcCommandBase {
 public:
  // Need to override this to serialize the RPC. This should destructively
  // create a message for the RPC (Hence the &&).
  c10::intrusive_ptr<Message> toMessage() && {
    JitRRefPickleGuard jitPickleGuard;
    return std::move(*this).toMessageImpl();
  }
  virtual c10::intrusive_ptr<Message> toMessageImpl() && = 0;
  virtual ~RpcCommandBase() = 0;
};

inline RpcCommandBase::~RpcCommandBase() = default;

inline c10::Dict<std::string, std::string> deviceMapToC10Dict(const DeviceMap& deviceMap) {
  c10::Dict<std::string, std::string> c10dict;
  for (const auto& mapEntry : deviceMap) {
    c10dict.insert(mapEntry.first.str(), mapEntry.second.str());
  }
  return c10dict;
}

inline DeviceMap c10DictToDeviceMap(const c10::Dict<std::string, std::string>& c10DeviceMap) {
  DeviceMap deviceMap;
  for (const auto& mapEntry : c10DeviceMap) {
    deviceMap.insert({mapEntry.key(), mapEntry.value()});
  }
  return deviceMap;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
