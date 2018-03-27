#ifndef CAFFE2_CORE_NET_H_
#define CAFFE2_CORE_NET_H_

#include <atomic>
#include <climits>
#include <cstddef>
#include <thread> // NOLINT
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator_schema.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/simple_queue.h"

namespace caffe2 {

class NetBase;
typedef ObserverBase<NetBase> NetObserver;
typedef std::function<std::unique_ptr<NetObserver>(NetBase*)>
    NetObserverCreator;

class OperatorBase;
class Workspace;

// Net is a thin struct that owns all the operators together with the operator
// contexts.
class NetBase : public Observable<NetBase> {
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
    for (const Event* event : events_) {
      if (event->Query() != EventStatus::EVENT_SUCCESS) {
        CAFFE_THROW(event->ErrorMessage());
      }
    }
    return true;
  }

  virtual bool RunAsync();

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
      const bool /*run_individual*/) {
    LOG(ERROR) << "Benchmark not implemented for this net type.";
    return vector<float>();
  }

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

  vector<string> external_input_;
  vector<string> external_output_;
  string name_;
  vector<const Event*> events_;
  std::shared_ptr<const NetDef> net_def_;
  DISABLE_COPY_AND_ASSIGN(NetBase);
};

CAFFE_DECLARE_REGISTRY(
    NetRegistry,
    NetBase,
    const std::shared_ptr<const NetDef>&,
    Workspace*);
#define REGISTER_NET_CREATOR(key, ...) \
  CAFFE_REGISTER_CREATOR(NetRegistry, key, __VA_ARGS__)
#define REGISTER_NET(name, ...) \
  CAFFE_REGISTER_CLASS(NetRegistry, name, __VA_ARGS__)

/**
 * @brief Creates a network, accessing / creating blobs in the given workspace.
 *
 * Note that this is different from Workspace::CreateNet. The latter adds the
 * created net object to the workspace's net map, while this function returns
 * a standalone net object.
 */
unique_ptr<NetBase> CreateNet(const NetDef& net_def, Workspace* ws);
unique_ptr<NetBase> CreateNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws);

void AddGlobalNetObserverCreator(NetObserverCreator creator);

} // namespace caffe2

#endif // CAFFE2_CORE_NET_H_
