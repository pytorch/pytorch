#include "Master.h"
#include "Master.hpp"
#include "State.hpp"
#include "../worker/Worker.h"
#include "../../process_group/General.hpp"


namespace thd {

namespace master {

std::unique_ptr<MasterCommandChannel> masterCommandChannel;
uint64_t tensorCount;

} // namespace master

bool THDMasterWorkerInit(THDChannelType channel_type) {
  if (!THDProcessGroupInit(channel_type)) return false;

  if (dataChannel->getRank() > 0) {
    THDWorkerMain();
  }

  // TODO: initialize master
  return true;
}

} // namespace thd
