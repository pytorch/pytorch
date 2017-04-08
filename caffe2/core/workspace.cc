#include "caffe2/core/workspace.h"

#include <algorithm>
#include <ctime>
#include <mutex>

#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2.pb.h"

CAFFE2_DEFINE_bool(
    caffe2_handle_executor_threads_exceptions,
    false,
    "If used we will handle exceptions in executor threads. "
    "This avoids SIGABRT but may cause process to deadlock");

CAFFE2_DEFINE_bool(
    caffe2_print_blob_sizes_at_exit,
    false,
    "If true, workspace destructor will print all blob shapes");

#if CAFFE2_MOBILE
// Threadpool restrictions

// Whether or not threadpool caps apply to Android
CAFFE2_DEFINE_int(caffe2_threadpool_android_cap, true, "");

// Whether or not threadpool caps apply to iOS
CAFFE2_DEFINE_int(caffe2_threadpool_ios_cap, false, "");

#endif // CAFFE2_MOBILE

namespace caffe2 {

namespace {
// try to get the should_stop signal, a scalar bool blob value.
// if the blob doesn't exist or is not initiaized, return false
inline const bool getShouldStop(const Blob* b) {
  if (!b || !b->meta().id()) { // not exist or uninitialized
    return false;
  }

  const auto& t = b->Get<TensorCPU>();
  CAFFE_ENFORCE(t.IsType<bool>() && t.size() == 1, "expects a scalar boolean");
  return *(t.template data<bool>());
}

// Returns a function that returns `true` if we should continue
// iterating, given the current iteration count.
std::function<bool(int64_t)> getContinuationTest(
    Workspace* ws,
    const ExecutionStep& step) {
  if (step.has_should_stop_blob()) {
    CAFFE_ENFORCE(
        !step.has_num_iter(),
        "Must not specify num_iter if should_stop_blob is set");
  }

  if (!step.has_should_stop_blob()) { // control by iteration
    CAFFE_ENFORCE(!step.has_only_once(), "not supported");
    int64_t iterations = step.has_num_iter() ? step.num_iter() : 1;
    VLOG(1) << "Will execute step " << step.name() << " for " << iterations
            << " iterations.";
    return [=](int64_t i) { return i < iterations; };
  } else { // control by signal blob
    bool onlyOnce = step.has_only_once() && step.only_once();
    VLOG(1) << "Will execute step" << step.name() << (onlyOnce ? " once " : "")
            << " until stopped by blob " << step.should_stop_blob();
    if (onlyOnce) {
      return [](int64_t i) { return i == 0; };
    } else {
      return [](int64_t i) { return true; };
    }
  }
};
}  // namespace

struct CompiledExecutionStep {
  typedef std::function<bool(int)> ShouldContinue;

  CompiledExecutionStep(
      const ExecutionStep* mainStep,
      Workspace* ws,
      ShouldContinue externalShouldContinue)
      : step(mainStep) {
    CAFFE_ENFORCE(
        (step->substep_size() == 0 || step->network_size() == 0),
        "An ExecutionStep should either have substep or networks"
        "but not both.");

    if (step->has_should_stop_blob()) {
      shouldStop = ws->GetBlob(step->should_stop_blob());
      CAFFE_ENFORCE(
          shouldStop, "blob ", step->should_stop_blob(), " does not exist");
    }

    if (step->substep_size()) {
      ShouldContinue substepShouldContinue;
      if (!step->concurrent_substeps() || step->substep().size() <= 1) {
        substepShouldContinue = externalShouldContinue;
      } else {
        substepShouldContinue = [this, externalShouldContinue](int64_t it) {
          return !gotFailure && externalShouldContinue(it);
        };
      }

      for (const auto& ss : step->substep()) {
        auto compiledSubstep = std::make_shared<CompiledExecutionStep>(
            &ss, ws, substepShouldContinue);
        if (ss.has_run_every_ms()) {
          reportSubsteps.push_back(compiledSubstep);
        } else {
          recurringSubsteps.push_back(compiledSubstep);
        }
      }
    } else {
      for (const string& network_name : step->network()) {
        auto* net = ws->GetNet(network_name);
        CAFFE_ENFORCE(net != nullptr, "Network ", network_name, " not found.");
        networks.push_back(net);
      }
    }

    netShouldContinue = getContinuationTest(ws, *step);
    shouldContinue = [this, externalShouldContinue](int64_t iter) {
      return externalShouldContinue(iter) && this->netShouldContinue(iter);
    };
  }

  const ExecutionStep* step;
  vector<std::shared_ptr<CompiledExecutionStep>> reportSubsteps;
  vector<std::shared_ptr<CompiledExecutionStep>> recurringSubsteps;

  vector<NetBase*> networks;
  Blob* shouldStop{nullptr};
  ShouldContinue netShouldContinue;
  ShouldContinue shouldContinue;
  std::atomic<bool> gotFailure{false};
};

void Workspace::PrintBlobSizes() {
  vector<string> blobs = LocalBlobs();
  size_t cumtotal = 0;

  // First get total sizes and sort
  vector<std::pair<size_t, std::string>> blob_sizes;
  for (const auto& s : blobs) {
    Blob* b = this->GetBlob(s);
    ShapeCall shape_fun = GetShapeCallFunction(b->meta().id());
    if (shape_fun) {
      bool shares_data = false;
      size_t capacity;
      auto shape = shape_fun(b->GetRaw(), shares_data, capacity);
      if (shares_data) {
        // Blobs sharing data do not actually take any memory
        capacity = 0;
      }
      cumtotal += capacity;
      blob_sizes.push_back(make_pair(capacity, s));
    }
  }
  std::sort(
      blob_sizes.begin(),
      blob_sizes.end(),
      [](const std::pair<size_t, std::string>& a,
         const std::pair<size_t, std::string>& b) {
        return b.first < a.first;
      });

  // Then print in descending order
  LOG(INFO) << "---- Workspace blobs: ---- ";
  LOG(INFO) << "name;current shape;capacity bytes;percentage";
  for (const auto& sb : blob_sizes) {
    Blob* b = this->GetBlob(sb.second);
    ShapeCall shape_fun = GetShapeCallFunction(b->meta().id());
    CHECK(shape_fun != nullptr);
    bool _shares_data = false;
    size_t capacity;
    auto shape = shape_fun(b->GetRaw(), _shares_data, capacity);
    std::stringstream ss;
    ss << sb.second << ";";
    for (const auto d : shape) {
      ss << d << ",";
    }
    LOG(INFO) << ss.str() << ";" << sb.first << ";" << std::setprecision(3)
              << (cumtotal > 0 ? 100.0 * double(sb.first) / cumtotal : 0.0)
              << "%";
  }
  LOG(INFO) << "Total;;" << cumtotal << ";100%";
}

vector<string> Workspace::LocalBlobs() const {
  vector<string> names;
  for (auto& entry : blob_map_) {
    names.push_back(entry.first);
  }
  return names;
}

vector<string> Workspace::Blobs() const {
  vector<string> names;
  for (auto& entry : blob_map_) {
    names.push_back(entry.first);
  }
  if (shared_) {
    vector<string> shared_blobs = shared_->Blobs();
    names.insert(names.end(), shared_blobs.begin(), shared_blobs.end());
  }
  return names;
}

Blob* Workspace::CreateBlob(const string& name) {
  if (HasBlob(name)) {
    VLOG(1) << "Blob " << name << " already exists. Skipping.";
  } else {
    VLOG(1) << "Creating blob " << name;
    blob_map_[name] = unique_ptr<Blob>(new Blob());
  }
  return GetBlob(name);
}

bool Workspace::RemoveBlob(const string& name) {
  auto it = blob_map_.find(name);
  if (it != blob_map_.end()) {
    VLOG(1) << "Removing blob " << name << " from this workspace.";
    blob_map_.erase(it);
    return true;
  }

  // won't go into share_ here
  VLOG(1) << "Blob " << name << " not exists. Skipping.";
  return false;
}

const Blob* Workspace::GetBlob(const string& name) const {
  if (blob_map_.count(name)) {
    return blob_map_.at(name).get();
  } else if (shared_ && shared_->HasBlob(name)) {
    return shared_->GetBlob(name);
  } else {
    LOG(WARNING) << "Blob " << name << " not in the workspace.";
    // TODO(Yangqing): do we want to always print out the list of blobs here?
    // LOG(WARNING) << "Current blobs:";
    // for (const auto& entry : blob_map_) {
    //   LOG(WARNING) << entry.first;
    // }
    return nullptr;
  }
}

Blob* Workspace::GetBlob(const string& name) {
  return const_cast<Blob*>(
      static_cast<const Workspace*>(this)->GetBlob(name));
}

NetBase* Workspace::CreateNet(const NetDef& net_def) {
  CAFFE_ENFORCE(net_def.has_name(), "Net definition should have a name.");
  if (net_map_.count(net_def.name()) > 0) {
    LOG(WARNING) << "Overwriting existing network of the same name.";
    // Note(Yangqing): Why do we explicitly erase it here? Some components of
    // the old network, such as a opened LevelDB, may prevent us from creating a
    // new network before the old one is deleted. Thus we will need to first
    // erase the old one before the new one can be constructed.
    net_map_.erase(net_def.name());
  }
  // Create a new net with its name.
  VLOG(1) << "Initializing network " << net_def.name();
  net_map_[net_def.name()] =
      unique_ptr<NetBase>(caffe2::CreateNet(net_def, this));
  if (net_map_[net_def.name()].get() == nullptr) {
    LOG(ERROR) << "Error when creating the network.";
    net_map_.erase(net_def.name());
    return nullptr;
  }
  return net_map_[net_def.name()].get();
}

NetBase* Workspace::GetNet(const string& name) {
  if (!net_map_.count(name)) {
    return nullptr;
  } else {
    return net_map_[name].get();
  }
}

void Workspace::DeleteNet(const string& name) {
  if (net_map_.count(name)) {
    net_map_.erase(name);
  }
}

bool Workspace::RunNet(const string& name) {
  if (!net_map_.count(name)) {
    LOG(ERROR) << "Network " << name << " does not exist yet.";
    return false;
  }
  return net_map_[name]->Run();
}

bool Workspace::RunOperatorOnce(const OperatorDef& op_def) {
  std::unique_ptr<OperatorBase> op(CreateOperator(op_def, this));
  if (op.get() == nullptr) {
    LOG(ERROR) << "Cannot create operator of type " << op_def.type();
    return false;
  }
  if (!op->Run()) {
    LOG(ERROR) << "Error when running operator " << op_def.type();
    return false;
  }
  return true;
}
bool Workspace::RunNetOnce(const NetDef& net_def) {
  std::unique_ptr<NetBase> net(caffe2::CreateNet(net_def, this));
  if (!net->Run()) {
    LOG(ERROR) << "Error when running network " << net_def.name();
    return false;
  }
  return true;
}

bool Workspace::RunPlan(const PlanDef& plan,
                        ShouldContinue shouldContinue) {
  LOG(INFO) << "Started executing plan.";
  if (plan.execution_step_size() == 0) {
    LOG(WARNING) << "Nothing to run - did you define a correct plan?";
    // We will do nothing, but the plan is still legal so we will return true.
    return true;
  }
  LOG(INFO) << "Initializing networks.";

  for (const NetDef& net_def : plan.network()) {
    if (!CreateNet(net_def)) {
      LOG(ERROR) << "Failed initializing the networks.";
      return false;
    }
  }
  Timer plan_timer;
  for (const ExecutionStep& step : plan.execution_step()) {
    Timer step_timer;
    CompiledExecutionStep compiledStep(&step, this, shouldContinue);
    if (!ExecuteStepRecursive(compiledStep)) {
      LOG(ERROR) << "Failed initializing step " << step.name();
      return false;
    }
    LOG(INFO) << "Step " << step.name() << " took " << step_timer.Seconds()
                   << " seconds.";
  }
  LOG(INFO) << "Total plan took " << plan_timer.Seconds() << " seconds.";
  LOG(INFO) << "Plan executed successfully.";
  return true;
}

#if CAFFE2_MOBILE
ThreadPool* Workspace::GetThreadPool() {
  std::lock_guard<std::mutex> guard(thread_pool_creation_mutex_);

  if (!thread_pool_) {
    int numThreads = std::thread::hardware_concurrency();

    bool applyCap = false;
#if CAFFE2_ANDROID
    applyCap = caffe2::FLAGS_caffe2_threadpool_android_cap;
#elif CAFFE2_IOS
    applyCap = caffe2::FLAGS_caffe2_threadpool_ios_cap;
#else
#error Undefined architecture
#endif

    if (applyCap) {
      // 1 core  -> 1 thread
      // 2 cores -> 2 threads
      // 4 cores -> 3 threads
      // 8 cores -> 4 threads
      // more, continue limiting to half of available cores

      if (numThreads <= 3) {
        // no change
      } else if (numThreads <= 5) {
        // limit to 3
        numThreads = 3;
      } else {
        // Use half the cores
        numThreads = numThreads / 2;
      }
    }

    LOG(INFO) << "Constructing thread pool with " << numThreads << " threads";
    thread_pool_.reset(new ThreadPool(numThreads));
  }

  return thread_pool_.get();
}
#endif // CAFFE2_MOBILE

namespace {

struct Reporter {
  struct ReporterInstance {
    std::mutex report_mutex;
    std::condition_variable report_cv;
    std::thread report_thread;
    ReporterInstance(int intervalMillis, bool* done, std::function<void()> f) {
      auto interval = std::chrono::milliseconds(intervalMillis);
      auto reportWorker = [=]() {
        std::unique_lock<std::mutex> lk(report_mutex);
        do {
          report_cv.wait_for(lk, interval, [&]() { return *done; });
          f();
        } while (!*done);
      };
      report_thread = std::thread(reportWorker);
    }
  };

  void start(int64_t intervalMillis, std::function<void()> f) {
    instances_.emplace_back(new ReporterInstance(intervalMillis, &done, f));
  }

  ~Reporter() {
    done = true;
    for (auto& instance : instances_) {
      if (!instance->report_thread.joinable()) {
        continue;
      }
      instance->report_cv.notify_all();
      instance->report_thread.join();
    }
  }

 private:
  std::vector<std::unique_ptr<ReporterInstance>> instances_;
  bool done{false};
};

}

#define CHECK_SHOULD_STOP(step, shouldStop)                       \
  if (getShouldStop(shouldStop)) {                                \
    VLOG(1) << "Execution step " << step.name() << " stopped by " \
            << step.should_stop_blob();                           \
    return true;                                                  \
  }

bool Workspace::ExecuteStepRecursive(CompiledExecutionStep& compiledStep) {
  const auto& step = *(compiledStep.step);
  VLOG(1) << "Running execution step " << step.name();

  std::unique_ptr<Reporter> reporter;
  if (step.has_report_net() || compiledStep.reportSubsteps.size() > 0) {
    reporter = caffe2::make_unique<Reporter>();
    if (step.has_report_net()) {
      CAFFE_ENFORCE(
          step.has_report_interval(),
          "A report_interval must be provided if report_net is set.");
      if (net_map_.count(step.report_net()) == 0) {
        LOG(ERROR) << "Report net " << step.report_net() << " not found.";
      }
      VLOG(1) << "Starting reporter net";
      auto* net = net_map_[step.report_net()].get();
      reporter->start(step.report_interval() * 1000, [=]() {
        if (!net->Run()) {
          LOG(WARNING) << "Error running report_net.";
        }
      });
    }
    for (auto& compiledSubstep : compiledStep.reportSubsteps) {
      reporter->start(compiledSubstep->step->run_every_ms(), [=]() {
        if (!ExecuteStepRecursive(*compiledSubstep)) {
          LOG(WARNING) << "Error running report step.";
        }
      });
    }
  }

  const Blob* shouldStop = compiledStep.shouldStop;

  if (step.substep_size()) {
    bool sequential = !step.concurrent_substeps() || step.substep().size() <= 1;
    for (int64_t iter = 0; compiledStep.shouldContinue(iter); ++iter) {
      if (sequential) {
        VLOG(1) << "Executing step " << step.name() << " iteration " << iter;
        for (auto& compiledSubstep : compiledStep.recurringSubsteps) {
          if (!ExecuteStepRecursive(*compiledSubstep)) {
            return false;
          }
          CHECK_SHOULD_STOP(step, shouldStop);
        }
      } else {
        VLOG(1) << "Executing step " << step.name() << " iteration " << iter
                << " with " << step.substep().size() << " concurrent substeps";

        std::atomic<int> next_substep{0};
        std::mutex exception_mutex;
        string first_exception;
        auto worker = [&]() {
          while (true) {
            int substep_id = next_substep++;
            if (compiledStep.gotFailure ||
                (substep_id >= compiledStep.recurringSubsteps.size())) {
              break;
            }
            try {
              if (!ExecuteStepRecursive(
                      *compiledStep.recurringSubsteps.at(substep_id))) {
                compiledStep.gotFailure = true;
              }
            } catch (const std::exception& ex) {
              std::lock_guard<std::mutex> guard(exception_mutex);
              if (!first_exception.size()) {
                first_exception = GetExceptionString(ex);
                LOG(ERROR) << "Parallel worker exception:\n" << first_exception;
              }
              compiledStep.gotFailure = true;
              if (!FLAGS_caffe2_handle_executor_threads_exceptions) {
                // In complex plans other threads might get stuck if another
                // one fails. So we let exception to go out of thread which
                // causes SIGABRT. In local setup one might use this flag
                // in order to use Python debugger after a failure
                throw;
              }
            }
          }
        };

        std::vector<std::thread> threads;
        for (int64_t i = 0; i < step.substep().size(); ++i) {
          if (!step.substep().Get(i).has_run_every_ms()) {
            threads.emplace_back(worker);
          }
        }
        for (auto& thread: threads) {
          thread.join();
        }
        if (compiledStep.gotFailure) {
          LOG(ERROR) << "One of the workers failed.";
          if (first_exception.size()) {
            CAFFE_THROW(
                "One of the workers died with an unhandled exception ",
                first_exception);
          }
          return false;
        }
        // concurrent substeps should be careful about setting should_stop_blob
        CHECK_SHOULD_STOP(step, shouldStop);
      }
    }
    return true;
  } else {
    // If this ExecutionStep just contains nets, we can directly run it.
    for (int64_t iter = 0; compiledStep.shouldContinue(iter); ++iter) {
      VLOG(1) << "Executing networks " << step.name() << " iteration " << iter;
      for (NetBase* network : compiledStep.networks) {
        if (!network->Run()) {
          return false;
        }
        CHECK_SHOULD_STOP(step, shouldStop);
      }
    }
  }
  return true;
}

#undef CHECK_SHOULD_STOP

}  // namespace caffe2
