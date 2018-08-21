#pragma once

#include <future>

#include "caffe2/core/workspace.h"

namespace caffe2 {
namespace python {

class BackgroundPlan {
 public:
  BackgroundPlan(Workspace* ws, PlanDef def) : ws_(ws), def_(def) {}

  void run() {
    fut_ =
        std::async(std::launch::async, [this]() { return ws_->RunPlan(def_); });
  }

  bool isDone() {
    CAFFE_ENFORCE(fut_.valid());
    auto status = fut_.wait_for(std::chrono::milliseconds(0));
    return status == std::future_status::ready;
  }

  bool isSucceeded() {
    CAFFE_ENFORCE(isDone());
    return fut_.get();
  }

 private:
  Workspace* ws_;
  PlanDef def_;

  std::future<bool> fut_;
};

} // namespace python
} // namespace caffe2
