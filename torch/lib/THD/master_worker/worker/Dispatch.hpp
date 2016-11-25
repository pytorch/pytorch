#pragma once

#include "../common/RPC.hpp"

#include <memory>

namespace thd {
namespace worker {

std::string execute(std::unique_ptr<rpc::RPCMessage> raw_message_ptr);

} // namespace worker
} // namespace thd
