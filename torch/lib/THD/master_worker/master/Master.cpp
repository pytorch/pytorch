#include "Master.h"
#include "../worker/Worker.h"
#include "../../process_group/General.hpp"


namespace thd {

bool THDMasterWorkerInit(THDChannelType channel_type) {
  if (!THDProcessGroupInit(channel_type)) return false;

  if (dataChannel->getRank() > 0) {
    THDWorkerMain();
  }

  // TODO: initialize master
  return true;
}

} // namespace thd
