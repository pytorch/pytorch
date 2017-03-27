#include "../../process_group/General.hpp"
#include <THPP/Storage.hpp>
#include <THPP/Tensor.hpp>
#include "../common/RPC.hpp"
#include "Dispatch.hpp"
#include "Worker.h"
#include "Worker.hpp"

#include <stdexcept>

namespace thd {
namespace worker {

std::unique_ptr<WorkerCommandChannel> workerCommandChannel;
std::unordered_map<object_id_type, std::unique_ptr<thpp::Tensor>> workerTensors;
std::unordered_map<object_id_type, std::unique_ptr<thpp::Storage>> workerStorages;

} // namespace worker
} // namespace thd

using namespace thd;

void THDWorkerMain() {
  std::unique_ptr<thd::rpc::RPCMessage> command;
  for (;;) {
    command = worker::workerCommandChannel->recvMessage();
    auto msg = worker::execute(std::move(command));
    if (msg != "")
      fprintf(stderr, "WORKER %d: %s\n", (int)dataChannel->getRank(), msg.c_str());
  }
}
