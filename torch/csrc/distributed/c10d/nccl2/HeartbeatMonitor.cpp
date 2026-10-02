// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/HeartbeatMonitor.hpp>

#include <c10/util/Logging.h>
#include <c10/util/thread_name.h>
#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCLLazy.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace c10d::nccl2 {

using Clock = std::chrono::steady_clock;

struct Heartbeat {
  Heartbeat(std::string log_prefix, int global_rank)
      : log_prefix(std::move(log_prefix)), global_rank(global_rank) {}

  std::atomic<uint64_t> beats{0};
  const std::string log_prefix;
  const int global_rank;
  // Monitor thread only.
  uint64_t seen_beats{0};
  Clock::time_point seen_at{Clock::now()};
  bool reported{false};
};

namespace {

constexpr auto kPollInterval = std::chrono::seconds(1);

// One thread for the whole process: every group dumps into the same flight
// recorder file and the debug pipe is per process, as in stock where only the
// uid 0 group's monitor does either.
class HeartbeatMonitor {
 public:
  static HeartbeatMonitor& get() {
    // Leaked: the detached thread may run during static destruction.
    static auto* monitor = new HeartbeatMonitor();
    return *monitor;
  }

  void add(const std::shared_ptr<Heartbeat>& heartbeat) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!started_) {
      // Detached so that no destructor ever waits on it.
      try {
        std::thread([this]() { run(); }).detach();
        started_ = true;
      } catch (const std::exception& e) {
        LOG(ERROR) << "[TC] Cannot start the heartbeat monitor: " << e.what();
      }
    }
    heartbeats_.insert(heartbeat);
    cv_.notify_one();
  }

  // Never waits on the monitor thread, so destroying a group is not delayed.
  void remove(const std::shared_ptr<Heartbeat>& heartbeat) {
    std::lock_guard<std::mutex> lock(mutex_);
    heartbeats_.erase(heartbeat);
  }

 private:
  HeartbeatMonitor()
      : timeout_(getCvarInt(::c10d::TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC, 60 * 8)),
        terminate_(getCvarBool(::c10d::TORCH_NCCL_ENABLE_MONITORING, true)),
        log_cpp_stack_(getCvarBool(
            ::c10d::TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN,
            true)),
        dump_on_timeout_(getCvarBool(::c10d::TORCH_FR_DUMP_ON_TIMEOUT, true)),
        dump_wait_(
            getCvarInt(::c10d::TORCH_FR_WAIT_TIMEOUT_DUMP_MILSEC, 15 * 1000)),
        pipe_file_(getCvarString({"TORCH_NCCL_DEBUG_INFO_PIPE_FILE"}, "")) {
    LOG(INFO) << "[TC] nccl2 HeartbeatMonitor environments: "
              << "TORCH_NCCL_ENABLE_MONITORING: " << terminate_
              << ", TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: " << timeout_.count()
              << ", TORCH_FR_DUMP_ON_TIMEOUT: " << dump_on_timeout_
              << ", TORCH_FR_WAIT_TIMEOUT_DUMP_MILSEC: " << dump_wait_.count()
              << ", TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: "
              << log_cpp_stack_;
  }

  void run() {
    c10::setThreadName("pt_nccl2_heartbt");
    std::optional<::c10d::DumpPipe> pipe;
    bool pipe_opened = false;
    std::vector<std::future<void>> pipe_dumps;
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      cv_.wait(lock, [this]() { return !heartbeats_.empty(); });
      cv_.wait_for(lock, kPollInterval);
      const auto now = Clock::now();
      int pipe_rank = -1;
      std::shared_ptr<Heartbeat> stalled;
      for (const auto& heartbeat : heartbeats_) {
        if (pipe_rank < 0) {
          pipe_rank = heartbeat->global_rank;
        }
        const auto beats = heartbeat->beats.load(std::memory_order_relaxed);
        if (beats != heartbeat->seen_beats) {
          heartbeat->seen_beats = beats;
          heartbeat->seen_at = now;
          heartbeat->reported = false;
          continue;
        }
        if (stalled || heartbeat->reported || timeout_.count() <= 0 ||
            now - heartbeat->seen_at < timeout_) {
          continue;
        }
        heartbeat->reported = true;
        stalled = heartbeat;
      }
      lock.unlock();

      if (!pipe_opened && pipe_rank >= 0) {
        pipe_opened = true;
        try {
          pipe.emplace(
              pipe_rank,
              pipe_file_,
              static_cast<int>(
                  getFlightRecorder(std::string(ProcessGroupNCCL::kBackendName))
                      ->max_entries_));
        } catch (const std::exception& e) {
          LOG(ERROR) << "[TC] Cannot open the debug info pipe: " << e.what();
        }
      }
      if (pipe && pipe->shouldDump()) {
        LOG(INFO) << "[TC] Dump signal received through pipe, triggering FR "
                     "dump.";
        // Best effort and not waited on, as in stock.
        std::erase_if(pipe_dumps, [](const std::future<void>& f) {
          return f.wait_for(std::chrono::seconds(0)) ==
              std::future_status::ready;
        });
        const bool only_active =
            getCvarBool(::c10d::TORCH_INCLUDE_ONLY_ACTIVE, false);
        try {
          pipe_dumps.push_back(std::async(std::launch::async, [only_active]() {
            dumpTrace(/*include_stack_traces=*/true, only_active);
          }));
        } catch (const std::exception& e) {
          LOG(ERROR) << "[TC] Cannot start flight recorder dump: " << e.what();
        }
      }
      if (stalled) {
        handleStall(stalled);
      }
      lock.lock();
    }
  }

  static void dumpTrace(bool include_stack_traces, bool only_active) {
    try {
      for (const auto backend :
           {ProcessGroupNCCL::kBackendName,
            ProcessGroupNCCLLazy::kBackendName}) {
        if (::c10d::try_dump_fr_trace_file(
                /*includeCollectives=*/true,
                include_stack_traces,
                only_active,
                std::string(backend))) {
          return;
        }
      }
      LOG(ERROR) << "[TC] No flight recorder trace to dump.";
    } catch (const std::exception& e) {
      LOG(ERROR) << "[TC] Flight recorder dump failed: " << e.what();
    }
  }

  // Bounded like FlightRecorderHook::onAbort: retry without stack traces,
  // whose symbolization may need the GIL, and leak attempts that time out
  // since ~future would join them.
  void dumpOnStall(const std::string& log_prefix) {
    if (!dump_on_timeout_ || dumped_) {
      return;
    }
    dumped_ = true;
    bool include_stack_traces =
        getCvarBool(::c10d::TORCH_INCLUDE_STACK_TRACE, true);
    const bool only_active =
        getCvarBool(::c10d::TORCH_INCLUDE_ONLY_ACTIVE, false);
    while (true) {
      std::future<void> dump;
      try {
        dump = std::async(
            std::launch::async, [include_stack_traces, only_active]() {
              dumpTrace(include_stack_traces, only_active);
            });
      } catch (const std::exception& e) {
        LOG(ERROR) << log_prefix
                   << "Cannot start flight recorder dump: " << e.what();
        return;
      }
      if (dump.wait_for(dump_wait_) == std::future_status::ready) {
        return;
      }
      abandoned_.push_back(std::move(dump));
      LOG(ERROR) << log_prefix << "Flight recorder dump did not finish within "
                 << dump_wait_.count() << " ms.";
      if (!include_stack_traces) {
        return;
      }
      include_stack_traces = false;
    }
  }

  void handleStall(const std::shared_ptr<Heartbeat>& heartbeat) {
    const auto& prefix = heartbeat->log_prefix;
    LOG(ERROR)
        << prefix << "ProcessGroupNCCL's watchdog got stuck for "
        << timeout_.count()
        << " seconds without making progress in monitoring enqueued collectives. "
        << "This typically indicates a NCCL/CUDA API (e.g., CudaEventDestroy) hang blocking the watchdog, "
        << "and could be triggered by another thread holding the GIL inside a "
        << "CUDA api (for example, CudaEventDestroy), or other deadlock-prone behaviors. "
        << "If you suspect the watchdog is not actually stuck and a longer timeout would help, "
        << "you can either increase the timeout (TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC) to a larger value "
        << "or disable the heartbeat monitor (TORCH_NCCL_ENABLE_MONITORING=0).";

    dumpOnStall(prefix);

    if (::c10d::get_gil_checker() != nullptr) {
      // Detached: it blocks for as long as the GIL is held.
      auto promise = std::make_shared<std::promise<bool>>();
      auto gil = promise->get_future();
      try {
        std::thread([promise]() {
          c10::setThreadName("pt_nccl2_gil_chk");
          try {
            promise->set_value((*::c10d::get_gil_checker())());
          } catch (...) {
            promise->set_exception(std::current_exception());
          }
        }).detach();
        if (gil.wait_for(std::chrono::milliseconds(300)) !=
            std::future_status::ready) {
          LOG(ERROR) << prefix
                     << "Could not acquire GIL within 300 ms on exit, possible "
                        "GIL induced hang";
        }
      } catch (const std::exception& e) {
        LOG(ERROR) << prefix << "Cannot check the GIL: " << e.what();
      }
    }

    auto& cpp_dumper = ::c10d::get_cpp_trace_dumper();
    if (log_cpp_stack_ && cpp_dumper.has_value()) {
      LOG(INFO) << prefix << "Dumping c++ stacktraces:";
      cpp_dumper.value()(
          [&](const std::string& line) { LOG(INFO) << prefix << line; });
      LOG(INFO) << prefix << "Finished c++ stacktraces dump.";
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (heartbeats_.count(heartbeat) == 0 ||
          heartbeat->beats.load(std::memory_order_relaxed) !=
              heartbeat->seen_beats) {
        LOG(INFO) << prefix
                  << "Watchdog recovered or exited while dumping debug info.";
        return;
      }
    }
    constexpr const char* kReason =
        "after attempting to dump debug info, due to ProcessGroupNCCL watchdog hang.";
    if (terminate_) {
      LOG(FATAL) << prefix << "Terminating the process " << kReason;
    }
    LOG(ERROR) << prefix
               << "ProcessGroupNCCL monitor thread is disabled, but would have "
                  "terminated the process "
               << kReason;
  }

  const std::chrono::seconds timeout_;
  const bool terminate_;
  const bool log_cpp_stack_;
  const bool dump_on_timeout_;
  const std::chrono::milliseconds dump_wait_;
  const std::string pipe_file_;

  // Leaf lock: never held while dumping or terminating.
  std::mutex mutex_;
  std::condition_variable cv_;
  std::unordered_set<std::shared_ptr<Heartbeat>> heartbeats_;
  bool started_ = false;

  // Monitor thread only.
  bool dumped_ = false;
  std::vector<std::future<void>> abandoned_;
};

} // namespace

HeartbeatRegistration::HeartbeatRegistration(
    std::string log_prefix,
    int global_rank)
    : heartbeat_(
          std::make_shared<Heartbeat>(std::move(log_prefix), global_rank)) {
  HeartbeatMonitor::get().add(heartbeat_);
}

HeartbeatRegistration::~HeartbeatRegistration() {
  HeartbeatMonitor::get().remove(heartbeat_);
}

void HeartbeatRegistration::beat() {
  heartbeat_->beats.fetch_add(1, std::memory_order_relaxed);
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
