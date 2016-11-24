#include "../../base/Tensor.hpp"
#include "Dispatch.hpp"
#include "../common/RPC.hpp"
#include "Worker.h"
#include "Worker.hpp"

#include <memory>
#include <unordered_map>

namespace thd {
namespace worker {

std::unique_ptr<WorkerCommandChannel> workerCommandChannel;
std::unordered_map<unsigned long long, Tensor*> tensors;

} // namespace worker
} // namespace thd

void THDWorkerMain() {
  // TODO: initialize worker
  std::unique_ptr<thd::rpc::RPCMessage> command;
  for (;;) {
    command = thd::worker::workerCommandChannel->recvMessage();
    thd::worker::execute(std::move(command));
  }
}
