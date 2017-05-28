#include "Master.h"
#include "Master.hpp"
#include "State.hpp"
#include "../worker/Worker.h"
#include "../../process_group/General.hpp"
#include "../../base/Exceptions.hpp"

namespace thd {
namespace master {

std::unique_ptr<MasterCommandChannel> masterCommandChannel;

} // namespace master
} // namespace thd

using namespace thd;
using namespace thd::master;

void THDMasterWorkerInit(THDChannelType channel_type, std::string init_method,
                         int world_size, std::string group_name) {
  HANDLE_EXCEPTIONS
  THDProcessGroupInit(channel_type, init_method, world_size, group_name);

  if (dataChannel->getRank() > 0) {
    /*
     * Worker initialization. It goes into infinite loop in which waits
     * for commands from master. Returning from `THDWorkerMain` indicates
     * a failure so it will `return false`.
     */
    THDWorkerMain(init_method, world_size, group_name);
    THError("unexpected exit from worker main loop");
  }

  THDState::s_workers = std::vector<WorkerState>(dataChannel->getNumProcesses());

  InitMethod::Config config = getInitConfig(init_method, world_size, group_name);
  masterCommandChannel.reset(new MasterCommandChannel(config));
  masterCommandChannel->init();

  END_HANDLE_EXCEPTIONS
}
