#pragma once

#include "../common/RPC.hpp"

#include <memory>

namespace thd {
namespace worker {

void execute(std::unique_ptr<rpc::RPCMessage> raw_message_ptr);

} // namespace worker
} // namespace thd
