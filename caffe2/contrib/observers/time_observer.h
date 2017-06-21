#ifndef CAFFE2_CONTRIB_OBSERVERS_TIME_OBSERVER_H_
#define CAFFE2_CONTRIB_OBSERVERS_TIME_OBSERVER_H_

#include <unordered_map>

#include "caffe2/core/common.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"

namespace caffe2 {

template <class T>
class TimeObserver final : public ObserverBase<T> {
 public:
  explicit TimeObserver<T>(T* subject_) : ObserverBase<T>(subject_) {}
  inline float average_time() const {
    return total_time_ / iterations_;
  }
  float average_time_children() const {
    float sum = 0.0f;
    for (auto& ob : children_) {
      sum += ob.get()->average_time();
    }
    return sum / children_.size();
  }
  ~TimeObserver() {}

 private:
  Timer timer_;
  float start_time_ = 0.0f;
  float total_time_ = 0.0f;
  int iterations_ = 0;

  vector<unique_ptr<TimeObserver<OperatorBase>>> children_;

  bool Start() override;
  bool Stop() override;
};

} // namespace caffe2

#endif // CAFFE2_CONTRIB_OBSERVERS_TIME_OBSERVER_H_
