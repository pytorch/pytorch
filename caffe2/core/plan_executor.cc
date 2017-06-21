#include "caffe2/core/plan_executor.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "caffe2/core/timer.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"

CAFFE2_DEFINE_bool(
    caffe2_handle_executor_threads_exceptions,
    false,
    "If used we will handle exceptions in executor threads. "
    "This avoids SIGABRT but may cause process to deadlock");

namespace caffe2 {

namespace {

struct NetDefInfo {
  const NetDef* netDef;
  // in order to keep the "override existing nets" on the top-level workflow,
  // we need to makr the nets that already exist so that we can override them
  // exactly once.
  bool needsOverride;
};

using NetDefMap = std::unordered_map<std::string, NetDefInfo>;

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

// if the blob doesn't exist or is not initiaized, return false
inline const bool getShouldStop(const Blob* b) {
  if (!b || !b->meta().id()) { // not exist or uninitialized
    return false;
  }

  const auto& t = b->Get<TensorCPU>();
  CAFFE_ENFORCE(t.IsType<bool>() && t.size() == 1, "expects a scalar boolean");
  return *(t.template data<bool>());
}

struct CompiledExecutionStep;

/**
 * Controls compilation and runtime cloning of execution steps.
 *
 * If step.create_workspace=False, this wrapper will compile the execution step
 * and its children once, and calls to ExecutionStepWrapper::compiled() will
 * always return the same compiled step.
 * If step.create_workspace=True, no compilation is done at creation time.
 * Instead, a new CompiledExecutionStep is created for every compiled() call.
 *
 * CompiledExecutionStep owns its Workspace, and the lifetime of the
 * compiled step along with its workspace will be tied to the lifetime of
 * the `CompileGuard` object returned by compiled().
 *
 * ExecuteStepRecursive will call call compiled() once before the given
 * execution step is run and keep it alive for the length of its execution.
 * This means that, for steps with create_workspace=true, a child workspace
 * will be created everytime the step is executed, and destroyed right
 * afterwards.
 */
struct ExecutionStepWrapper {
  ExecutionStepWrapper(
      const ExecutionStep* step,
      Workspace* externalWorkspace,
      ShouldContinue externalShouldContinue,
      NetDefMap* netDefs)
      : step_(step),
        externalWorkspace_(externalWorkspace),
        externalShouldContinue_(externalShouldContinue),
        netDefs_(netDefs) {
    // If this execution step does not create a child workspace,
    // then just eagerly-compile it. This will trigger CreateNet on the
    // nets used by this execution step.
    if (!step_->create_workspace()) {
      compiledStep_ = doCompile();
    }
  }

  class CompiledGuard {
    void reset(std::unique_ptr<CompiledExecutionStep>&& compiled) {
      compiled_ = std::move(compiled);
      compiledRef_ = compiled_.get();
    }
    void reset(CompiledExecutionStep* compiledRef) {
      compiled_.reset();
      compiledRef_ = compiledRef;
    }

   public:
    CompiledExecutionStep* operator->() {
      return compiledRef_;
    }

   private:
    CompiledGuard() {}
    std::unique_ptr<CompiledExecutionStep> compiled_;
    CompiledExecutionStep* compiledRef_;
    friend class ExecutionStepWrapper;
  };

  const ExecutionStep& step() {
    return *step_;
  }

  CompiledGuard compiled() {
    CompiledGuard guard;
    if (compiledStep_) {
      guard.reset(compiledStep_.get());
    } else {
      guard.reset(doCompile());
    }
    return guard;
  }

 private:
  std::unique_ptr<CompiledExecutionStep> doCompile();

  const ExecutionStep* step_;
  Workspace* externalWorkspace_;
  ShouldContinue externalShouldContinue_;
  NetDefMap* netDefs_;
  std::unique_ptr<CompiledExecutionStep> compiledStep_;
};

struct CompiledExecutionStep {
  typedef std::function<bool(int)> ShouldContinue;

  CompiledExecutionStep(
      const ExecutionStep* mainStep,
      Workspace* externalWorkspace,
      ShouldContinue externalShouldContinue,
      NetDefMap* netDefs)
      : step(mainStep) {
    if (mainStep->create_workspace()) {
      localWorkspace_.reset(new Workspace(externalWorkspace));
      workspace = localWorkspace_.get();
    } else {
      workspace = externalWorkspace;
    }

    CAFFE_ENFORCE(
        (step->substep_size() == 0 || step->network_size() == 0),
        "An ExecutionStep should either have substep or networks"
        "but not both.");

    auto createAndGetNet = [&](const std::string& network_name) {
      auto it = netDefs->find(network_name);
      CAFFE_ENFORCE(
          it != netDefs->end(),
          "ExecutionStep " + mainStep->name() + " uses undefined net " +
              network_name);
      // needsOverride does not need synchronization because it is only
      // relevant for non-dynamic executions steps. This is due to the fact
      // that concurrent nets run on child workspaces, that do not needOverride.
      if (it->second.needsOverride || !workspace->GetNet(network_name)) {
        workspace->CreateNet(*it->second.netDef, true);
        it->second.needsOverride = false;
      }
      auto* net = workspace->GetNet(network_name);
      CAFFE_ENFORCE(net != nullptr, "Network ", network_name, " not found.");
      return net;
    };

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
        auto compiledSubstep = std::make_shared<ExecutionStepWrapper>(
            &ss, workspace, substepShouldContinue, netDefs);
        if (ss.has_run_every_ms()) {
          reportSubsteps.push_back(compiledSubstep);
        } else {
          recurringSubsteps.push_back(compiledSubstep);
        }
      }
    } else {
      for (const string& network_name : step->network()) {
        networks.push_back(createAndGetNet(network_name));
      }
    }

    if (step->has_should_stop_blob()) {
      shouldStop = workspace->GetBlob(step->should_stop_blob());
      CAFFE_ENFORCE(
          shouldStop, "blob ", step->should_stop_blob(), " does not exist");
    }

    if (step->has_report_net()) {
      CAFFE_ENFORCE(
          step->has_report_interval(),
          "A report_interval must be provided if report_net is set.");
      reportNet = createAndGetNet(step->report_net());
    } else {
      reportNet = nullptr;
    }

    netShouldContinue = getContinuationTest(workspace, *step);
    shouldContinue = [this, externalShouldContinue](int64_t iter) {
      return externalShouldContinue(iter) && this->netShouldContinue(iter);
    };
  }

  const ExecutionStep* step;
  Workspace* workspace;
  vector<std::shared_ptr<ExecutionStepWrapper>> reportSubsteps;
  vector<std::shared_ptr<ExecutionStepWrapper>> recurringSubsteps;

  vector<NetBase*> networks;
  NetBase* reportNet;
  Blob* shouldStop{nullptr};
  ShouldContinue netShouldContinue;
  ShouldContinue shouldContinue;
  std::atomic<bool> gotFailure{false};

 private:
  std::unique_ptr<Workspace> localWorkspace_;
};

std::unique_ptr<CompiledExecutionStep> ExecutionStepWrapper::doCompile() {
  return std::unique_ptr<CompiledExecutionStep>(new CompiledExecutionStep(
      step_, externalWorkspace_, externalShouldContinue_, netDefs_));
}

#define CHECK_SHOULD_STOP(step, shouldStop)                       \
  if (getShouldStop(shouldStop)) {                                \
    VLOG(1) << "Execution step " << step.name() << " stopped by " \
            << step.should_stop_blob();                           \
    return true;                                                  \
  }

bool ExecuteStepRecursive(ExecutionStepWrapper& stepWrapper) {
  const auto& step = stepWrapper.step();
  auto compiledStep = stepWrapper.compiled();

  VLOG(1) << "Running execution step " << step.name();

  std::unique_ptr<Reporter> reporter;
  if (step.has_report_net() || compiledStep->reportSubsteps.size() > 0) {
    reporter = caffe2::make_unique<Reporter>();
    auto* reportNet = compiledStep->reportNet;
    if (reportNet) {
      VLOG(1) << "Starting reporter net";
      reporter->start(step.report_interval() * 1000, [reportNet]() {
        if (!reportNet->Run()) {
          LOG(WARNING) << "Error running report_net.";
        }
      });
    }
    for (auto& substepWrapper : compiledStep->reportSubsteps) {
      reporter->start(
          substepWrapper->step().run_every_ms(), [substepWrapper]() {
            if (!ExecuteStepRecursive(*substepWrapper)) {
              LOG(WARNING) << "Error running report step.";
            }
          });
    }
  }

  const Blob* shouldStop = compiledStep->shouldStop;

  if (step.substep_size()) {
    bool sequential =
        (!step.concurrent_substeps() || step.substep().size() <= 1) &&
        (!step.has_num_concurrent_instances() ||
         step.num_concurrent_instances() <= 1);
    for (int64_t iter = 0; compiledStep->shouldContinue(iter); ++iter) {
      if (sequential) {
        VLOG(1) << "Executing step " << step.name() << " iteration " << iter;
        for (auto& substepWrapper : compiledStep->recurringSubsteps) {
          if (!ExecuteStepRecursive(*substepWrapper)) {
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
          auto num_substeps = compiledStep->recurringSubsteps.size();
          int substep_id = next_substep++ % num_substeps;
          if (compiledStep->gotFailure) {
            return;
          }
          try {
            if (!ExecuteStepRecursive(
                    *compiledStep->recurringSubsteps.at(substep_id))) {
              compiledStep->gotFailure = true;
            }
          } catch (const std::exception& ex) {
            std::lock_guard<std::mutex> guard(exception_mutex);
            if (!first_exception.size()) {
              first_exception = GetExceptionString(ex);
              LOG(ERROR) << "Parallel worker exception:\n" << first_exception;
            }
            compiledStep->gotFailure = true;
            if (!FLAGS_caffe2_handle_executor_threads_exceptions) {
              // In complex plans other threads might get stuck if another
              // one fails. So we let exception to go out of thread which
              // causes SIGABRT. In local setup one might use this flag
              // in order to use Python debugger after a failure
              throw;
            }
          }
        };

        std::vector<std::thread> threads;
        auto numThreads = compiledStep->recurringSubsteps.size();
        if (step.has_num_concurrent_instances()) {
          numThreads *= step.num_concurrent_instances();
        }
        for (int64_t i = 0; i < numThreads; ++i) {
          threads.emplace_back(worker);
        }
        for (auto& thread : threads) {
          thread.join();
        }
        if (compiledStep->gotFailure) {
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
    for (int64_t iter = 0; compiledStep->shouldContinue(iter); ++iter) {
      VLOG(1) << "Executing networks " << step.name() << " iteration " << iter;
      for (NetBase* network : compiledStep->networks) {
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
}

bool RunPlanOnWorkspace(
    Workspace* ws,
    const PlanDef& plan,
    ShouldContinue shouldContinue) {
  LOG(INFO) << "Started executing plan.";
  if (plan.execution_step_size() == 0) {
    LOG(WARNING) << "Nothing to run - did you define a correct plan?";
    // We will do nothing, but the plan is still legal so we will return true.
    return true;
  }
  LOG(INFO) << "Initializing networks.";

  NetDefMap net_defs;
  for (const NetDef& net_def : plan.network()) {
    CAFFE_ENFORCE(
        net_defs.count(net_def.name()) == 0,
        "Your plan contains networks of the same name \"",
        net_def.name(),
        "\", which should not happen. Check your plan to see "
        "if you made a programming error in creating the plan.");
    auto netAlreadyExists = ws->GetNet(net_def.name()) != nullptr;
    net_defs[net_def.name()] = NetDefInfo{&net_def, netAlreadyExists};
  }
  Timer plan_timer;
  for (const ExecutionStep& step : plan.execution_step()) {
    Timer step_timer;
    ExecutionStepWrapper stepWrapper(&step, ws, shouldContinue, &net_defs);
    if (!ExecuteStepRecursive(stepWrapper)) {
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
}
