#ifndef _THD_STATE_H
#define _THD_STATE_H

#include <vector>

namespace thd {
namespace master {

class WorkerState {
};

class THDState {
  static std::vector<WorkerState> workers;
  thread_local static size_t current_worker;
};

} // namespace master
} // namespace thd

#endif
