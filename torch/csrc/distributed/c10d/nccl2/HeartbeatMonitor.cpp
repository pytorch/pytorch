// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/HeartbeatMonitor.hpp>

#include <c10/util/Logging.h>
#include <c10/util/thread_name.h>
#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/nccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCLLazy.hpp>
#include <torch/csrc/jit/serialization/pickle.h>

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
  explicit Heartbeat(std::string log_prefix)
      : log_prefix(std::move(log_prefix)) {}

  std::atomic<uint64_t> beats{0};
  const std::string log_prefix;
  // Monitor thread only.
  uint64_t seen_beats{0};
  Clock::time_point seen_at{Clock::now()};
  bool reported{false};
  // Set once the stall is reported: terminate if still stalled by then.
  std::optional<Clock::time_point> terminate_at;
};

namespace {

constexpr auto kPollInterval = std::chrono::seconds(1);
constexpr std::string_view kBackends[] = {
    ProcessGroupNCCL::kBackendName,
    ProcessGroupNCCLLazy::kBackendName};

// Global rank FlightRecorderHook set from the groups' rank mappings, which
// also names the dump file; -1 until a group is attached.
int recorderRank() {
  for (const auto backend : kBackends) {
    const int rank = getFlightRecorder(std::string(backend))->getRank();
    if (rank >= 0) {
      return rank;
    }
  }
  return -1;
}

// Interleaves entries b into a by creation time. b's record ids come from
// another recorder's counter, so each b entry takes the id of the a entry
// before it (or the first a entry): the FR analyzer aligns ranks on the first
// record id and drops lower ones (align_trace_from_beginning).
c10::List<c10::IValue> mergeEntries(
    const c10::List<c10::IValue>& a,
    const c10::List<c10::IValue>& b) {
  if (a.empty()) {
    return b;
  }
  auto created = [](const c10::IValue& entry) {
    return entry.toGenericDict().at(::c10d::time_created_key).toInt();
  };
  c10::List<c10::IValue> merged(a.elementType());
  merged.reserve(a.size() + b.size());
  c10::IValue record_id = a.get(0).toGenericDict().at(::c10d::record_id_key);
  size_t j = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const c10::IValue entry = a.get(i);
    for (; j < b.size() && created(b.get(j)) < created(entry); ++j) {
      b.get(j).toGenericDict().insert_or_assign(
          ::c10d::record_id_key, record_id);
      merged.push_back(b.get(j));
    }
    record_id = entry.toGenericDict().at(::c10d::record_id_key);
    merged.push_back(entry);
  }
  for (; j < b.size(); ++j) {
    b.get(j).toGenericDict().insert_or_assign(::c10d::record_id_key, record_id);
    merged.push_back(b.get(j));
  }
  return merged;
}

// Merges trace b into a: dicts by key, entries by mergeEntries.
void mergeTrace(
    const c10::Dict<c10::IValue, c10::IValue>& a,
    const c10::Dict<c10::IValue, c10::IValue>& b) {
  for (const auto& item : b) {
    auto it = a.find(item.key());
    if (it == a.end()) {
      a.insert(item.key(), item.value());
      continue;
    }
    const auto value = it->value();
    if (value.isGenericDict() && item.value().isGenericDict()) {
      auto dict = value.toGenericDict();
      for (const auto& e : item.value().toGenericDict()) {
        dict.insert_or_assign(e.key(), e.value());
      }
      continue;
    }
    if (item.key() == ::c10d::entries_key && value.isList() &&
        item.value().isList()) {
      a.insert_or_assign(
          item.key(), mergeEntries(value.toList(), item.value().toList()));
    }
  }
}

// One thread for the whole process: the nccl2 and nccl-lazy recorders are
// merged into one dump file and the debug pipe is per process, as in stock
// where only the uid 0 group's monitor does either.
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
        TC_LOG(ERROR) << "Cannot start the heartbeat monitor: " << e.what();
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
        include_stack_traces_(
            getCvarBool(::c10d::TORCH_INCLUDE_STACK_TRACE, true)),
        only_active_(getCvarBool(::c10d::TORCH_INCLUDE_ONLY_ACTIVE, false)),
        pipe_file_(getCvarString({"TORCH_NCCL_DEBUG_INFO_PIPE_FILE"}, "")) {
    TC_LOG(INFO) << "nccl2 HeartbeatMonitor environments: "
                 << "TORCH_NCCL_ENABLE_MONITORING: " << terminate_
                 << ", TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: " << timeout_.count()
                 << ", TORCH_FR_DUMP_ON_TIMEOUT: " << dump_on_timeout_
                 << ", TORCH_FR_WAIT_TIMEOUT_DUMP_MILSEC: "
                 << dump_wait_.count()
                 << ", TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: "
                 << log_cpp_stack_;
  }

  void run() {
    c10::setThreadName("pt_nccl2_hbmon");
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      cv_.wait(lock, [this]() { return !heartbeats_.empty(); });
      cv_.wait_for(lock, kPollInterval);
      lock.unlock();
      // Before the scan, so that a termination acts on fresh state.
      pollPipe();
      lock.lock();
      const auto now = Clock::now();
      std::shared_ptr<Heartbeat> stalled;
      std::shared_ptr<Heartbeat> hung;
      for (const auto& heartbeat : heartbeats_) {
        const auto beats = heartbeat->beats.load(std::memory_order_relaxed);
        if (beats != heartbeat->seen_beats) {
          if (heartbeat->reported) {
            LOG(INFO) << heartbeat->log_prefix
                      << "Watchdog recovered, not terminating the process.";
          }
          heartbeat->seen_beats = beats;
          heartbeat->seen_at = now;
          heartbeat->reported = false;
          heartbeat->terminate_at.reset();
          continue;
        }
        if (heartbeat->terminate_at && now >= *heartbeat->terminate_at) {
          hung = heartbeat;
          continue;
        }
        if (heartbeat->reported || timeout_.count() <= 0 ||
            now - heartbeat->seen_at < timeout_) {
          continue;
        }
        if (!stalled || heartbeat->seen_at < stalled->seen_at) {
          stalled = heartbeat;
        }
      }
      lock.unlock();
      if (hung) {
        terminate(*hung);
      }
      if (stalled) {
        stalled->reported = true;
        handleStall(*stalled);
        // Stock waits another timeout after the dump before terminating.
        stalled->terminate_at = Clock::now() + timeout_;
      }
      lock.lock();
    }
  }

  void pollPipe() {
    // Waits for a group to publish the global rank, so that the pipe is
    // named like the dump file and never takes another process's.
    if (!pipe_opened_ && !pipe_file_.empty() && recorderRank() >= 0) {
      pipe_opened_ = true;
      try {
        pipe_.emplace(
            recorderRank(),
            pipe_file_,
            static_cast<int>(
                getFlightRecorder(std::string(ProcessGroupNCCL::kBackendName))
                    ->max_entries_));
      } catch (const std::exception& e) {
        TC_LOG(ERROR) << "Cannot open the debug info pipe: " << e.what();
      }
    }
    if (!pipe_ || !pipe_->shouldDump()) {
      return;
    }
    TC_LOG(INFO) << "Dump signal received through pipe, triggering FR dump.";
    // Best effort and not waited on, as in stock.
    std::erase_if(pipe_dumps_, [](const std::future<void>& f) {
      return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    });
    try {
      pipe_dumps_.push_back(std::async(std::launch::async, [this]() {
        dumpTrace(/*include_stack_traces=*/true, only_active_);
      }));
    } catch (const std::exception& e) {
      TC_LOG(ERROR) << "Cannot start flight recorder dump: " << e.what();
    }
  }

  // Writes one file holding every nccl2 and nccl-lazy group's entries: both
  // backends' recorders name the same <prefix><rank> file.
  static void dumpTrace(bool include_stack_traces, bool only_active) {
    try {
      const int rank = recorderRank();
      std::optional<c10::Dict<c10::IValue, c10::IValue>> merged;
      for (const auto backend : kBackends) {
        auto* recorder = getFlightRecorder(std::string(backend));
        if (!recorder->enabled_ || recorder->getRank() < 0) {
          continue;
        }
        const auto trace = ::c10d::dump_fr_trace(
            /*includeCollectives=*/true,
            include_stack_traces,
            only_active,
            std::string(backend));
        auto dict =
            torch::jit::unpickle(trace.data(), trace.size()).toGenericDict();
        if (!merged) {
          merged = std::move(dict);
          continue;
        }
        mergeTrace(*merged, dict);
      }
      if (!merged) {
        TC_LOG(ERROR) << "No flight recorder trace to dump.";
        return;
      }
      ::c10d::write_fr_trace_file(rank, ::c10d::pickle_str(*merged));
    } catch (const std::exception& e) {
      TC_LOG(ERROR) << "Flight recorder dump failed: " << e.what();
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
    bool include_stack_traces = include_stack_traces_;
    while (true) {
      std::future<void> dump;
      try {
        dump = std::async(std::launch::async, [this, include_stack_traces]() {
          dumpTrace(include_stack_traces, only_active_);
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

  void handleStall(const Heartbeat& heartbeat) {
    const auto& prefix = heartbeat.log_prefix;
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
          c10::setThreadName("pt_nccl2_gilchk");
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
  }

  void terminate(Heartbeat& heartbeat) {
    const auto& prefix = heartbeat.log_prefix;
    heartbeat.terminate_at.reset();
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
  const bool include_stack_traces_;
  const bool only_active_;
  const std::string pipe_file_;

  // Leaf lock: never held while dumping or terminating.
  std::mutex mutex_;
  std::condition_variable cv_;
  std::unordered_set<std::shared_ptr<Heartbeat>> heartbeats_;
  bool started_ = false;

  // Monitor thread only.
  std::optional<::c10d::DumpPipe> pipe_;
  bool pipe_opened_ = false;
  std::vector<std::future<void>> pipe_dumps_;
  bool dumped_ = false;
  std::vector<std::future<void>> abandoned_;
};

} // namespace

void initHeartbeatMonitor() {
  HeartbeatMonitor::get();
}

HeartbeatRegistration::HeartbeatRegistration(std::string log_prefix)
    : heartbeat_(std::make_shared<Heartbeat>(std::move(log_prefix))) {
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
