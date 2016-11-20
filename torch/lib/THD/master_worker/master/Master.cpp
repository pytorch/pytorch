#include "../../base/DataChannel.hpp"
#include "../worker/Worker.h"

namespace thd {

bool THDInit() {
  // TODO: initialize registry with all available backends
  auto dataChannel = DataChannelRegistry::dataChannelFor(0);
  if (!dataChannel->init()) {
    return false;
  }

  if (dataChannel->getRank() > 0) {
    THDWorkerMain();
  }

  // TODO: initialize master
  return true;
}

} // namespace thd
