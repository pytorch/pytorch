#pragma once

#include "caffe2/core/net.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

class RunCountNetObserver;
class RunCountOperatorObserver final : public ObserverBase<OperatorBase> {
 public:
  explicit RunCountOperatorObserver(OperatorBase* op) = delete;
  RunCountOperatorObserver(OperatorBase* op, RunCountNetObserver* netObserver);
  ~RunCountOperatorObserver() {}

  std::unique_ptr<ObserverBase<OperatorBase>> copy(
      OperatorBase* subject) override;

 private:
  void Start() override;
  void Stop() override;

 private:
  RunCountNetObserver* netObserver_;
};

class RunCountNetObserver final : public ObserverBase<NetBase> {
 public:
  explicit RunCountNetObserver(NetBase* subject_)
      : ObserverBase<NetBase>(subject_), cnt_(0) {}
  ~RunCountNetObserver() {}

  std::string debugInfo() override;

  friend class RunCountOperatorObserver;

 private:
  void Start() override;
  void Stop() override;

 protected:
  std::atomic<int> cnt_;
};

} // namespace caffe2
