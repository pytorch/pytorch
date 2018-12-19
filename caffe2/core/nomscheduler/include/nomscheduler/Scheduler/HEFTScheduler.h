//===----------------------------------------------------------------------===//
//
// nomnigraph supports for task scheduling problems.
// HEFT-based implementation of scheduler.
//
// (Heterogeneous Earliest Finish Time)
// Original description:
// Performance-effective and low-complexity task scheduling
// for heterogeneous computing
// H. Topcuoglu; S. Hariri; Min-You Wu
// IEEE Transactions on Parallel and Distributed Systems 2002
//
//===----------------------------------------------------------------------===//

#ifndef NOM_SCHEDULER_HEFT_SCHEDULER_H
#define NOM_SCHEDULER_HEFT_SCHEDULER_H

#include "HEFTScheduler-Internal.h"
#include "Scheduler.h"

namespace nomscheduler {

class HEFTScheduler : Scheduler {
 public:
  virtual SchedulerOutput schedule(const SchedulerInput& input) override;

  // Expose a scheduling method that also returns an internal algorithm state
  // for unit testing purpose.
  std::pair<SchedulerOutput, heftscheduler::AlgorithmState> scheduleInternal(
      const SchedulerInput& input);
};

} // namespace nomscheduler

#endif // NOM_SCHEDULER_HEFT_SCHEDULER_H
