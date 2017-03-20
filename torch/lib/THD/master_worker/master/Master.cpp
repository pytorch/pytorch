#include "Master.h"
#include "Master.hpp"
#include "State.hpp"
#include "../worker/Worker.h"
#include "../../process_group/General.hpp"

#include <string>
#include <thread>

namespace thd {
namespace master {

std::unique_ptr<MasterCommandChannel> masterCommandChannel;
std::thread masterErrorThread;

void errorHandler() {
  for (;;) {
    auto error = masterCommandChannel->recvError();
    THDState::s_error = "Error (worker " + std::to_string(std::get<0>(error)) + "): " + std::get<1>(error);
  }
}

} // namespace master
} // namespace thd

using namespace thd;
using namespace thd::master;

bool THDMasterWorkerInit(THDChannelType channel_type) {
  if (!THDProcessGroupInit(channel_type)) return false;

  if (dataChannel->getRank() > 0) {
    // Worker initialization. Can fail at start but then goes into inifity loop.
    return THDWorkerMain();
  }

  // Master initilization
  masterCommandChannel.reset(new MasterCommandChannel());
  bool ok = masterCommandChannel->init();
  if (!ok) {
    return false;
  }

  // start error handling thread
  std::thread tmp_thread(errorHandler);
  std::swap(masterErrorThread, tmp_thread);

  return true;
}
