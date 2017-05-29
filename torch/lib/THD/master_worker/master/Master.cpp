#include "Master.h"
#include "Master.hpp"
#include "State.hpp"
#include "../worker/Worker.h"
#include "../../process_group/General.hpp"

namespace thd {
namespace master {

std::unique_ptr<MasterCommandChannel> masterCommandChannel;

} // namespace master
} // namespace thd

using namespace thd;
using namespace thd::master;

bool THDMasterWorkerInit(THDChannelType channel_type) {
  if (!THDProcessGroupInit(channel_type)) return false;

  if (dataChannel->getRank() > 0) {
    /*
     * Worker initialization. It goes into infinite loop in which waits
     * for commands from master. Returning from `THDWorkerMain` indicates
     * a failure so it will `return false`.
     */
    THDWorkerMain();
    return false;
  }

  THDState::s_workers = std::vector<WorkerState>(dataChannel->getNumProcesses());

  masterCommandChannel.reset(new MasterCommandChannel());
  if (!masterCommandChannel->init()) {
    return false;
  }

  return true;
}
