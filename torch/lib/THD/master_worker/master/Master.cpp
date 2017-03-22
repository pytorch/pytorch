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
    /* Worker initialization. Can fail at start but then goes into infinite loop
     * in which waits for commands from master.
     */
    THDWorkerMain();
    return false;
  }

  /* Master initialization. We need to make sure that all connections which
   * are created in `init` function are set up because only then we can start
   * `masterErrorThread`.
   */
  masterCommandChannel.reset(new MasterCommandChannel());
  if (!masterCommandChannel->init()) {
    return false;
  }

  return true;
}
