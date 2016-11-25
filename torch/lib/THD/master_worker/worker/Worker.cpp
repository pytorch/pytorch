#include "../../base/Tensor.hpp"
#include "Dispatch.hpp"
#include "../common/RPC.hpp"
#include "Worker.h"
#include "Worker.hpp"

namespace thd {
namespace worker {

std::unique_ptr<WorkerCommandChannel> workerCommandChannel;
std::unordered_map<unsigned long long, std::unique_ptr<Tensor>> workerTensors;

} // namespace worker
} // namespace thd

void THDWorkerMain() {
  // TODO: initialize worker
  thd::worker::workerCommandChannel =
    std::unique_ptr<thd::WorkerCommandChannel>(new thd::WorkerCommandChannel(1));
  std::unique_ptr<thd::rpc::RPCMessage> command;
  for (;;) {
    command = thd::worker::workerCommandChannel->recvMessage();
    thd::worker::execute(std::move(command));
  }
}
