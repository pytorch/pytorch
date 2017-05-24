#include "caffe2/core/plan_executor.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
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

struct CompiledExecutionStep {
  typedef std::function<bool(int)> ShouldContinue;

  CompiledExecutionStep(
      const ExecutionStep* mainStep,
      Workspace* ws,
      ShouldContinue externalShouldContinue)
      : workspace(ws), step(mainStep) {
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

  Workspace* workspace;
  const ExecutionStep* step;
  vector<std::shared_ptr<CompiledExecutionStep>> reportSubsteps;
  vector<std::shared_ptr<CompiledExecutionStep>> recurringSubsteps;

  vector<NetBase*> networks;
  Blob* shouldStop{nullptr};
  ShouldContinue netShouldContinue;
  ShouldContinue shouldContinue;
  std::atomic<bool> gotFailure{false};
};

#define CHECK_SHOULD_STOP(step, shouldStop)                       \
  if (getShouldStop(shouldStop)) {                                \
    VLOG(1) << "Execution step " << step.name() << " stopped by " \
            << step.should_stop_blob();                           \
    return true;                                                  \
  }

bool ExecuteStepRecursive(CompiledExecutionStep& compiledStep) {
  const auto& step = *(compiledStep.step);
  Workspace* ws = compiledStep.workspace;

  VLOG(1) << "Running execution step " << step.name();

  std::unique_ptr<Reporter> reporter;
  if (step.has_report_net() || compiledStep.reportSubsteps.size() > 0) {
    reporter = caffe2::make_unique<Reporter>();
    if (step.has_report_net()) {
      CAFFE_ENFORCE(
          step.has_report_interval(),
          "A report_interval must be provided if report_net is set.");
      auto* net = ws->GetNet(step.report_net());
      if (!net) {
        LOG(ERROR) << "Report net " << step.report_net() << " not found.";
      }
      VLOG(1) << "Starting reporter net";
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
        for (auto& thread : threads) {
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

  std::set<string> seen_net_names_in_plan;
  for (const NetDef& net_def : plan.network()) {
    CAFFE_ENFORCE(
        seen_net_names_in_plan.count(net_def.name()) == 0,
        "Your plan contains networks of the same name \"",
        net_def.name(),
        "\", which should not happen. Check your plan to see "
        "if you made a programming error in creating the plan.");
    seen_net_names_in_plan.insert(net_def.name());
    // TODO(jiayq): consider if we want to override the default choice of
    // overwriting the nets if exists. The rationale here is that, a plan
    // is considered a big end-to-end thing (like a whole training run) and
    // is similar to the old Caffe Solver. It is big enough that we want to
    // give it a full control over the current workspace.
    if (!ws->CreateNet(net_def, true)) {
      LOG(ERROR) << "Failed initializing the networks.";
      return false;
    }
  }
  Timer plan_timer;
  for (const ExecutionStep& step : plan.execution_step()) {
    Timer step_timer;
    CompiledExecutionStep compiledStep(&step, ws, shouldContinue);
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
}
