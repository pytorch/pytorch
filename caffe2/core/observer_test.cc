#include <gtest/gtest.h>
#include "c10/util/Registry.h"
#include "caffe2/core/common.h"
#include "caffe2/core/net.h"
#include "caffe2/core/net_simple.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/scope_guard.h"

namespace caffe2 {

namespace {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::atomic<int> counter;

template <class T>
class DummyObserver final : public ObserverBase<T> {
 public:
  explicit DummyObserver<T>(T* subject_) : ObserverBase<T>(subject_) {}
  void Start() override;
  void Stop() override;

  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~DummyObserver() override {}
};

template <>
void DummyObserver<NetBase>::Start() {
  vector<OperatorBase*> operators = subject_->GetOperators();
  for (auto& op : operators) {
    op->AttachObserver(std::make_unique<DummyObserver<OperatorBase>>(op));
  }
  counter.fetch_add(1000);
}

template <>
void DummyObserver<OperatorBase>::Start() {
  counter.fetch_add(100);
}

template <>
void DummyObserver<NetBase>::Stop() {
  counter.fetch_add(10);
}

template <>
void DummyObserver<OperatorBase>::Stop() {
  counter.fetch_add(1);
}

class ObsTestDummyOp final : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  bool Run(int /* unused */) override {
    StartAllObservers();
    StopAllObservers();
    return true;
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(ObsTestDummy, ObsTestDummyOp);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CUDA_OPERATOR(ObsTestDummy, ObsTestDummyOp);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ObsTestDummy)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});

unique_ptr<NetBase> CreateNetTestHelper(Workspace* ws, bool isDAG = false) {
  NetDef net_def;
  if (isDAG) {
    net_def.set_type("dag");
  }
  {
    auto& op = *(net_def.add_op());
    op.set_type("ObsTestDummy");
    op.add_input("in");
    op.add_output("hidden");
  }
  {
    auto& op = *(net_def.add_op());
    op.set_type("ObsTestDummy");
    op.add_input("hidden");
    op.add_output("out");
  }
  net_def.add_external_input("in");
  net_def.add_external_output("out");

  return CreateNet(net_def, ws);
}
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ObserverTest, TestNotify) {
  auto count_before = counter.load();
  Workspace ws;
  ws.CreateBlob("in");
  NetDef net_def;
  unique_ptr<NetBase> net(CreateNetTestHelper(&ws));
  EXPECT_EQ(caffe2::dynamic_cast_if_rtti<SimpleNet*>(net.get()), net.get());
  unique_ptr<DummyObserver<NetBase>> net_ob =
      make_unique<DummyObserver<NetBase>>(net.get());
  net.get()->AttachObserver(std::move(net_ob));
  net.get()->Run();
  auto count_after = counter.load();
  EXPECT_EQ(1212, count_after - count_before);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ObserverTest, TestUniqueMap) {
  auto count_before = counter.load();
  Workspace ws;
  ws.CreateBlob("in");
  NetDef net_def;
  unique_ptr<NetBase> net(CreateNetTestHelper(&ws));
  EXPECT_EQ(caffe2::dynamic_cast_if_rtti<SimpleNet*>(net.get()), net.get());
  unique_ptr<DummyObserver<NetBase>> net_ob =
      make_unique<DummyObserver<NetBase>>(net.get());
  auto* ref = net.get()->AttachObserver(std::move(net_ob));
  net.get()->Run();
  unique_ptr<Observable<NetBase>::Observer> test =
      net.get()->DetachObserver(ref);
  auto count_after = counter.load();
  EXPECT_EQ(1212, count_after - count_before);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ObserverTest, TestNotifyAfterDetach) {
  auto count_before = counter.load();
  Workspace ws;
  ws.CreateBlob("in");
  NetDef net_def;
  unique_ptr<NetBase> net(CreateNetTestHelper(&ws));
  unique_ptr<DummyObserver<NetBase>> net_ob =
      make_unique<DummyObserver<NetBase>>(net.get());
  auto* ob = net.get()->AttachObserver(std::move(net_ob));
  net.get()->DetachObserver(ob);
  net.get()->Run();
  auto count_after = counter.load();
  EXPECT_EQ(0, count_after - count_before);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ObserverTest, TestDAGNetBase) {
  auto count_before = counter.load();
  Workspace ws;
  ws.CreateBlob("in");
  NetDef net_def;
  unique_ptr<NetBase> net(CreateNetTestHelper(&ws, true));
  unique_ptr<DummyObserver<NetBase>> net_ob =
      make_unique<DummyObserver<NetBase>>(net.get());
  net.get()->AttachObserver(std::move(net_ob));
  net.get()->Run();
  auto count_after = counter.load();
  EXPECT_EQ(1212, count_after - count_before);
}

#if 0
// This test intermittently segfaults,
// see https://github.com/pytorch/pytorch/issues/9137
TEST(ObserverTest, TestMultipleNetBase) {
  Workspace ws;
  ws.CreateBlob("in");
  NetDef net_def;
  unique_ptr<NetBase> net(CreateNetTestHelper(&ws, true));
  EXPECT_EQ(caffe2::dynamic_cast_if_rtti<NetBase*>(net.get()), net.get());

  // There may be some default observers
  const size_t prev_num = net.get()->NumObservers();
  const int num_tests = 100;
  vector<const Observable<NetBase>::Observer*> observers;
  for (int i = 0; i < num_tests; ++i) {
    unique_ptr<DummyObserver<NetBase>> net_ob =
        make_unique<DummyObserver<NetBase>>(net.get());
    observers.emplace_back(net.get()->AttachObserver(std::move(net_ob)));
  }

  net.get()->Run();

  for (const auto& observer : observers) {
    net.get()->DetachObserver(observer);
  }

  EXPECT_EQ(net.get()->NumObservers(), prev_num);
}
#endif
} // namespace caffe2
