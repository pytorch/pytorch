#pragma once

#include <functional>

namespace caffe2 {

class Workspace;
class PlanDef;

typedef std::function<bool(int)> ShouldContinue;

bool RunPlanOnWorkspace(Workspace* ws, const PlanDef& plan, ShouldContinue);
}
