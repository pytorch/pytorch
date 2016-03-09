#ifndef CAFFE2_OPERATORS_PREFETCH_OP_H_
#define CAFFE2_OPERATORS_PREFETCH_OP_H_

#include <thread>  // NOLINT

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class PrefetchOperator;

namespace internal {
// We define a prefetch function so that the prefetch function can call virtual
// member functions of the prefetch operator.
template <class Context>
void PrefetchFunc(PrefetchOperator<Context>* op) {
  op->prefetch_success_ = op->Prefetch();
}
}

// PrefetchOperator is an operator that prefetches the next batch. It should
// almost always be used to read things from disk, so I am setting the input to
// zero blobs.
template <class Context>
class PrefetchOperator : public OperatorBase {
 public:
  PrefetchOperator(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws),
        device_context_(operator_def.device_option()),
        prefetch_success_(false) {
    device_context_.SwitchToDevice();
  }
  virtual ~PrefetchOperator() {}

  bool Run() override {
    device_context_.SwitchToDevice();
    if (prefetch_thread_ == nullptr) {
      CAFFE_VLOG(1) << "Starting a new prefetch thread.";
      prefetch_thread_.reset(
          new std::thread(
              internal::PrefetchFunc<Context>, this));
    }
    // Join the last prefetch thread.
    CAFFE_VLOG(1) << "Waiting for the prefetch thread.";
    prefetch_thread_->join();

    if (!prefetch_success_) {
      CAFFE_LOG_ERROR << "Prefetching failed.";
      return false;
    }
    CAFFE_VLOG(1) << "Copy prefetched result.";
    if (!CopyPrefetched()) {
      CAFFE_LOG_ERROR << "Error when copying prefetched data.";
      return false;
    }
    prefetch_success_ = false;
    CAFFE_VLOG(1) << "Starting a new prefetch thread.";
    prefetch_thread_.reset(
        new std::thread(
            internal::PrefetchFunc<Context>, this));
    return device_context_.FinishDeviceComputation();
  }

  // You will need to implement this instead of the Run function.
  virtual bool Prefetch() = 0;
  virtual bool CopyPrefetched() = 0;
  friend void internal::PrefetchFunc<Context>(
      PrefetchOperator*);

 protected:
  Context device_context_;
  unique_ptr<std::thread> prefetch_thread_;
  bool prefetch_success_;

  INPUT_OUTPUT_STATS(0, 0, 1, INT_MAX);
  DISABLE_COPY_AND_ASSIGN(PrefetchOperator);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_PREFETCH_OP_H_
