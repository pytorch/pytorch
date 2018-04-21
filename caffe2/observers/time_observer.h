#ifndef CAFFE2_CONTRIB_OBSERVERS_TIME_OBSERVER_H_
#define CAFFE2_CONTRIB_OBSERVERS_TIME_OBSERVER_H_

#include <unordered_map>

#include "caffe2/core/common.h"
#include "caffe2/core/net.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/observers/operator_attaching_net_observer.h"
#include "caffe2/operators/rnn/rnn_capable_operator_observer.h"

namespace caffe2 {

class TimeObserver;
class TimeCounter {
 public:
  explicit TimeCounter() {}
  inline float average_time() const {
    return total_time_ / iterations_;
  }

 protected:
  Timer timer_;
  float start_time_ = 0.0f;
  float total_time_ = 0.0f;
  int iterations_ = 0;
};

class TimeOperatorObserver final : public TimeCounter,
                                   public RNNCapableOperatorObserver {
 public:
  explicit TimeOperatorObserver(OperatorBase* subject) = delete;
  explicit TimeOperatorObserver(
      OperatorBase* subject,
      TimeObserver* /* unused */)
      : RNNCapableOperatorObserver(subject) {}
  std::unique_ptr<ObserverBase<OperatorBase>> rnnCopy(
      OperatorBase* subject,
      int rnn_order) const override;

 private:
  void Start() override;
  void Stop() override;
};

class TimeObserver final
    : public TimeCounter,
      public OperatorAttachingNetObserver<TimeOperatorObserver, TimeObserver> {
 public:
  explicit TimeObserver(NetBase* subject)
      : OperatorAttachingNetObserver<TimeOperatorObserver, TimeObserver>(
            subject,
            this) {}

  float average_time_children() const {
    float sum = 0.0f;
    for (const auto* observer : operator_observers_) {
      sum += observer->average_time();
    }
    return sum / subject_->GetOperators().size();
  }

 private:
  void Start() override;
  void Stop() override;
};

} // namespace caffe2

#endif // CAFFE2_CONTRIB_OBSERVERS_TIME_OBSERVER_H_
