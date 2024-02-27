#pragma once

#include "caffe2/contrib/gloo/common.h"
#include "caffe2/core/operator.h"

#include <gloo/algorithm.h>
#include <gloo/barrier_all_to_one.h>
#include <gloo/common/error.h>
#include <gloo/context.h>

namespace caffe2 {
namespace gloo {

template <class Context>
class BarrierOp final : public Operator<Context> {
 public:
  BarrierOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        ws_(ws),
        status_blob_(
            OperatorBase::GetSingleArgument<std::string>("status_blob", "")) {
    if (status_blob_ != "") {
      ws_->CreateBlob(status_blob_);
    }
  }

  ~BarrierOp() override {}

  bool RunOnDevice() override {
    auto context = OperatorBase::Input<std::shared_ptr<::gloo::Context>>(0);
    std::call_once(once_, [&] {
      initContext_ = context;
      // Use an all-to-one barrier synchronizing against rank 0
      algorithm_.reset(new ::gloo::BarrierAllToOne(initContext_, 0));
    });

    // If any parameter has changed in between runs, the initialized
    // algorithm is invalid and cannot be used.
    CAFFE_ENFORCE(context == initContext_, "Context has changed");

    try {
      algorithm_->run();
    } catch (::gloo::IoException& ioe) {
      LOG(ERROR) << "Caught gloo IO exception: " << ioe.what();
      if (status_blob_ != "") {
        signalFailure(ws_->GetBlob(status_blob_), ioe);
        return false;
      } else {
        throw;
      }
    }
    return true;
  }

 protected:
  std::once_flag once_;
  std::shared_ptr<::gloo::Context> initContext_;
  std::unique_ptr<::gloo::Algorithm> algorithm_;
  Workspace* ws_;
  std::string status_blob_;
};
} // namespace gloo
} // namespace caffe2
