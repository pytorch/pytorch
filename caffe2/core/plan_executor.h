#pragma once

#include <functional>
#ifndef CAFFE2_MOBILE
#include "caffe2/core/stats.h"
#endif // CAFFE2_MOBILE

namespace caffe2 {

class Workspace;
class PlanDef;

typedef std::function<bool(int)> ShouldContinue;

bool RunPlanOnWorkspace(Workspace* ws, const PlanDef& plan, ShouldContinue);

#ifndef CAFFE2_MOBILE
struct PlanExecutionTime {
  CAFFE_STAT_CTOR(PlanExecutionTime);
  CAFFE_EXPORTED_STAT(plan_execution_time_ns);
};
#endif // CAFFE2_MOBILE
}
