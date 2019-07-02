#ifndef CAFFE2_CORE_NET_H_
#define CAFFE2_CORE_NET_H_

#include <atomic>
#include <climits>
#include <cstddef>
#include <thread> // NOLINT
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "c10/core/thread_pool.h"
#include "c10/util/Registry.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator_schema.h"
#include "caffe2/core/tensor.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/simple_queue.h"

C10_DECLARE_string(caffe2_override_executor);

namespace caffe2 {

class NetBase;
typedef ObserverBase<NetBase> NetObserver;
typedef std::function<std::unique_ptr<NetObserver>(NetBase*)>
    NetObserverCreator;

class OperatorBase;
class Workspace;

// Net is a thin struct that owns all the operators together with the operator
// contexts.
class CAFFE2_API NetBase : public Observable<NetBase> {
 public:
  NetBase(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  virtual ~NetBase() noexcept {}

  virtual bool SupportsAsync() = 0;
  inline const vector<const Event*>& events() const {
    return events_;
  }

  virtual void Wait() {
    // by default just wait till all events are finished
    for (const auto& event : events_) {
      event->Finish();
    }
  }

  virtual bool Run() {
    if (!RunAsync()) {
      LOG(ERROR) << "Failed to execute async run";
      return false;
    }
    Wait();
    return handleRunError();
  }

  virtual bool RunAsync();

  /* Benchmarks a network for one individual run so that we can feed new
   * inputs on additional calls.
   * This function returns the number of microseconds spent
   * during the benchmark
   */
  virtual float TEST_Benchmark_One_Run();

  /**
   * Benchmarks a network.
   *
   * This function returns a vector of float recording the number of milli-
   * seconds spent during the benchmark. The 0-th item is the time spent per
   * each network run, and if a net instantiation supports run_individual,
   * the remainder of the vector returns the number of milliseconds spent per
   * opeartor.
   */
  virtual vector<float> TEST_Benchmark(
      const int /*warmup_runs*/,
      const int /*main_runs*/,
      const bool /*run_individual*/);

  inline const vector<string>& external_output() const {
    return external_output_;
  }

  inline const vector<string>& external_input() const {
    return external_input_;
  }

  /* Used to attach Observers to operators of a Net
   *
   * Returns pointers to objects owned with unique_ptrs.
   * Use with caution.
   */
  virtual vector<OperatorBase*> GetOperators() const = 0;

  const string& Name() const {
    return name_;
  }

  inline const NetDef& debug_def() const {
    CAFFE_ENFORCE(has_debug_def(), "net_def was null!");
    return *net_def_;
  }

  inline bool has_debug_def() const {
    return net_def_ != nullptr;
  }

 protected:
  virtual bool DoRunAsync() {
    CAFFE_THROW("Not implemented");
  };

  virtual bool handleRunError() {
    for (const Event* event : events_) {
      if (event->Query() != EventStatus::EVENT_SUCCESS) {
        CAFFE_THROW(event->ErrorMessage());
      }
    }
    return true;
  }

  vector<string> external_input_;
  vector<string> external_output_;
  string name_;
  vector<const Event*> events_;
  std::shared_ptr<const NetDef> net_def_;
  C10_DISABLE_COPY_AND_ASSIGN(NetBase);
};

class CAFFE2_API ExecutorHelper {
 public:
  ExecutorHelper() {}
  virtual TaskThreadPoolBase* GetPool(const DeviceOption& option) const;
  virtual std::vector<OperatorBase*> GetOperators() const;
  virtual int GetNumWorkers() const;
  virtual ~ExecutorHelper() {}
};

C10_DECLARE_REGISTRY(
    NetRegistry,
    NetBase,
    const std::shared_ptr<const NetDef>&,
    Workspace*);
#define REGISTER_NET_CREATOR(key, ...) \
  C10_REGISTER_CREATOR(NetRegistry, key, __VA_ARGS__)
#define REGISTER_NET(name, ...) \
  C10_REGISTER_CLASS(NetRegistry, name, __VA_ARGS__)

/**
 * @brief Creates a network, accessing / creating blobs in the given workspace.
 *
 * Note that this is different from Workspace::CreateNet. The latter adds the
 * created net object to the workspace's net map, while this function returns
 * a standalone net object.
 */
CAFFE2_API unique_ptr<NetBase> CreateNet(const NetDef& net_def, Workspace* ws);
CAFFE2_API unique_ptr<NetBase> CreateNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws);

CAFFE2_API void AddGlobalNetObserverCreator(NetObserverCreator creator);

CAFFE2_API void ClearGlobalNetObservers();

} // namespace caffe2

#endif // CAFFE2_CORE_NET_H_
