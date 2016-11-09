#include "../_THD.h"

namespace thd {

bool THDInit() {
  // TODO: initialize registry with all available backends
  auto &backend = DataChannelRegistry::backend_for(0);
  if (!backend.init()) return false;
  if (backend.get_id() > 0) {
    THDWorkerMain();
  }
  // TODO: initialize master
  return true;
}

}
