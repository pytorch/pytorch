#pragma once
#include <cstdio>
#include <cstdlib>

#include <memory>
#include <mutex>

#include <ATen/ATen.h>
#ifdef USE_C10D_NCCL
#include <ATen/cuda/CUDAEvent.h>
#endif // USE_C10D_NCCL
#include <c10/util/Exception.h>
#include <c10/util/WaitCounter.h>
#include <nlohmann/json.hpp>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <optional>

namespace c10d {

#define DEFINE_CONSTANT(name, value) \
  static c10::IValue name = value;   \
  static std::string name##_str = value;
// Update whenever changing contents or formatting of the dump
// (minor when adding fields, major when changing existing fields)
// Also update both JSON and Pickle dumps to make use of the newly defined
// field(s).
DEFINE_CONSTANT(version_val, "2.7")
DEFINE_CONSTANT(entries_key, "entries")
DEFINE_CONSTANT(nccl_comm_key, "nccl_comm_state")
DEFINE_CONSTANT(nccl_version_key, "nccl_version")
DEFINE_CONSTANT(version_key, "version")
DEFINE_CONSTANT(pg_config_key, "pg_config")
DEFINE_CONSTANT(pg_status_key, "pg_status")
DEFINE_CONSTANT(record_id_key, "record_id")
DEFINE_CONSTANT(pg_id_key, "pg_id")
DEFINE_CONSTANT(pg_name_key, "process_group")
DEFINE_CONSTANT(collective_seq_id_key, "collective_seq_id")
DEFINE_CONSTANT(p2p_seq_id_key, "p2p_seq_id")
DEFINE_CONSTANT(is_p2p_key, "is_p2p")
DEFINE_CONSTANT(op_id_key, "op_id")
DEFINE_CONSTANT(profiling_name_key, "profiling_name")
DEFINE_CONSTANT(input_sizes_key, "input_sizes")
DEFINE_CONSTANT(input_dtypes_key, "input_dtypes")
DEFINE_CONSTANT(output_sizes_key, "output_sizes")
DEFINE_CONSTANT(output_dtypes_key, "output_dtypes")
DEFINE_CONSTANT(time_created_key, "time_created_ns")
DEFINE_CONSTANT(duration_key, "duration_ms")
DEFINE_CONSTANT(timeout_key, "timeout_ms")
DEFINE_CONSTANT(frames_key, "frames")
DEFINE_CONSTANT(state_key, "state")
DEFINE_CONSTANT(line_key, "line")
DEFINE_CONSTANT(name_key, "name")
DEFINE_CONSTANT(filename_key, "filename")
DEFINE_CONSTANT(retired_key, "retired")
DEFINE_CONSTANT(time_discovered_started_key, "time_discovered_started_ns")
DEFINE_CONSTANT(time_discovered_completed_key, "time_discovered_completed_ns")
DEFINE_CONSTANT(completed_state, "completed")
DEFINE_CONSTANT(scheduled_state, "scheduled")
DEFINE_CONSTANT(started_state, "started")
#undef DEFINE_CONSTANT

// Write NCCL debug info to local disk or any storage users define.
// There are some constrains we set for the debug info writer:
// 1. The writer should only be registered once.
// 2. Once registered, users cannot change it including un-register.
// 3. It is recommended to register the customized writer in the trainer setup,
//    If users don't register before calling launchAsyncDebugDump, then users
//    lose the chance to register (and the default writer will be
//    auto-registered).
class TORCH_API DebugInfoWriter {
 public:
  virtual ~DebugInfoWriter() = default;
  virtual void write(const std::string& trace);
  static DebugInfoWriter& getWriter(int rank);
  static void registerWriter(std::unique_ptr<DebugInfoWriter> writer);
  virtual std::string getWriterTarget() {
    return filename_;
  }

 protected:
  DebugInfoWriter(const std::string& namePrefix, int rank) {
    filename_ = c10::str(namePrefix, rank);
  }
  std::string filename_;

 private:
  static std::unique_ptr<DebugInfoWriter> writer_;
  static std::atomic<bool> hasWriterRegistered_;
};

/* Helper used by work::getDuration() and flight recorder */
float getDurationFromEvent(c10::Event& startEvent, c10::Event& endEvent);

#ifdef USE_C10D_NCCL
/* Helper used by work::getDuration() and nccl flight recorder */
float getDurationFromEvent(
    at::cuda::CUDAEvent& ncclStartEvent,
    at::cuda::CUDAEvent& ncclEndEvent);
#endif // USE_C10D_NCCL

template <typename EventType>
struct FlightRecorder {
  static FlightRecorder<EventType>* get() {
    // intentionally leak on exit
    // because this will hold python state that may get destructed
    static FlightRecorder<EventType>* instance =
        new FlightRecorder<EventType>();
    return instance;
  }
  FlightRecorder() {
    max_entries_ = getCvarInt(
        {"TORCH_TRACE_BUFFER_SIZE", "TORCH_NCCL_TRACE_BUFFER_SIZE"}, 0);
    capture_cpp_stack_ = getCvarBool(
        {"TORCH_TRACE_CPP_STACK", "TORCH_NCCL_TRACE_CPP_STACK"}, false);
    enabled_ = max_entries_ > 0;
  }
  struct Entry {
    size_t id_; // incremented id in the trace buffer
                // used to figure out where in the circular entries
                // buffer this entry will be located to
                // update state information
    size_t pg_id_;
    std::tuple<std::string, std::string> pg_name_; // <group_name, group_desc>

    // collective_seq_id and p2p_seq_id refer to actual kernel launches (e.g. 1
    // per coalesced group).
    // collective_seq_id only increments for true collective operations (over
    // all ranks in the group). p2p_seq_id only increments over non-collective
    // operations in the group. op_id refers to logical operations (e.g. one per
    // op inside coalesced group)
    size_t collective_seq_id_;
    size_t p2p_seq_id_;
    size_t op_id_;
    std::string profiling_name_;

    std::shared_ptr<torch::CapturedTraceback> traceback_;
    // we borrow pointers to start_ and end_ so we can query the state
    // on reporting. However, once the event is completed, the call
    // to `complete` will clear these.
    EventType *start_, *end_;

    // timestamp when the entry was created, likely close to the time the work
    // was 'enqueued'- not necessarily started
    c10::time_t time_created_;

    // configured timeout for this entry
    c10::time_t timeout_ms_;

    // Is this a P2P event?
    bool isP2P_;

    std::optional<float> duration_;

    // timestamp when our CPU threads discovered that the kernel started.
    // will always be _after_ it actually started, and can be very late
    // if the watchdog thread got stuck on CUDA APIs.
    std::optional<c10::time_t> time_discovered_started_;

    // timestamp when our CPU threads discovered that the kernel completed.
    // will always be _after_ it actually complated, and can be the same time
    // as the discovery of the start if the watchdog thread is stuck on CUDA
    // APIs
    std::optional<c10::time_t> time_discovered_completed_;

    // size information for input/output tensors
    c10::SmallVector<int64_t, 4> input_dims_;
    std::vector<c10::ScalarType> input_dtypes_;
    c10::SmallVector<int64_t, 4> output_dims_;
    std::vector<c10::ScalarType> output_dtypes_;
    c10::SmallVector<int64_t, 8> sizes_; // flattened from inputs, outputs
    bool retired_ = false; // is this work entry no longer in the workMetaList_?
                           // a retired but not completed event has timed out

    // Returns the traceback of current entry, in string form.
    // Note: `getTraceback` invokes `torch::symbolize`, which may need to
    // acquire the GIL. If you don't want to block the current thread or take
    // the risk of a GIL deadlock, you can use an asynchronous calling mechanism
    // like std::async.
    std::string getTraceback() {
      torch::CapturedTraceback* traceback = traceback_.get();
      torch::SymbolizedTracebacks s_tbs = torch::symbolize({traceback});
      // We use 0 because we only have one traceback here.
      const auto& s_tb = s_tbs.tracebacks.at(0);
      std::stringstream oss;
      for (auto idx : c10::irange(s_tb.size())) {
        auto frame_id = s_tb[idx];
        const auto& frame = s_tbs.all_frames.at(frame_id);
        oss << "#" << idx << " " << frame.funcname << " from " << frame.filename
            << ":" << frame.lineno << '\n';
      }
      /* Resulted format is like:
        #0 all_reduce from pytorch/torch/distributed/distributed_c10d.py:2696
        #1 wrapper from pytorch/torch/distributed/c10d_logger.py:83
        #2 bar from /home/user/repro.py:15
        #3 foo from /home/user/repro.py:24
        #4 main from /home/user/repro.py:34
        #5 <module> from /home/user/repro.py:40
      */
      return oss.str();
    };
  };

  bool enabled_ = false;
  bool capture_cpp_stack_ = false;
  std::mutex mutex_;
  std::vector<Entry> entries_;
  size_t max_entries_ = 0;
  size_t next_ = 0;
  size_t id_ = 0;
  std::map<size_t, std::shared_ptr<ProcessGroupStatus>> all_pg_status_ = {};
  std::map<std::tuple<std::string, std::string>, std::vector<uint64_t>>
      pg_name_to_ranks_ = {};
  std::string nccl_version_;

  std::optional<size_t> record(
      size_t pg_id,
      const std::tuple<std::string, std::string>& pg_name,
      size_t collective_seq_id,
      size_t p2p_seq_id,
      size_t op_id,
      std::string profiling_name,
      const std::vector<at::Tensor>& inputs,
      const std::vector<at::Tensor>& outputs,
      EventType* start,
      EventType* end,
      std::chrono::milliseconds timeout_ms,
      std::shared_ptr<ProcessGroupStatus> pg_status,
      bool isP2P) {
    if (!enabled_) {
      return std::nullopt;
    }
    if (all_pg_status_.find(pg_id) == all_pg_status_.end()) {
      // Current pg_status is not in FR.
      all_pg_status_[pg_id] = std::move(pg_status);
    }
    auto traceback =
        torch::CapturedTraceback::gather(true, true, capture_cpp_stack_);
    std::lock_guard<std::mutex> guard(mutex_);

    auto te = Entry{
        id_,
        pg_id,
        pg_name,
        collective_seq_id,
        p2p_seq_id,
        op_id,
        std::move(profiling_name),
        std::move(traceback),
        start,
        end,
        c10::getTime(),
        timeout_ms.count(),
        isP2P,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        {},
        {},
        {},
        {},
        {},
        false};

    for (const auto& input : inputs) {
      c10::IntArrayRef sizes = input.sizes();
      te.input_dtypes_.push_back(input.dtype().toScalarType());
      te.input_dims_.push_back(static_cast<int64_t>(sizes.size()));
      te.sizes_.insert(te.sizes_.end(), sizes.begin(), sizes.end());
    }

    for (const auto& output : outputs) {
      c10::IntArrayRef sizes = output.sizes();
      te.output_dtypes_.push_back(output.dtype().toScalarType());
      te.output_dims_.push_back(static_cast<int64_t>(sizes.size()));
      te.sizes_.insert(te.sizes_.end(), sizes.begin(), sizes.end());
    }

    if (entries_.size() < max_entries_) {
      entries_.emplace_back(std::move(te));
    } else {
      entries_[next_++] = std::move(te);
      if (next_ == max_entries_) {
        next_ = 0;
      }
    }
    return id_++;
  };

  void record_pg_ranks(
      const std::tuple<std::string, std::string>& pg_name,
      std::vector<uint64_t> ranks) {
    if (!enabled_) {
      return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    pg_name_to_ranks_[pg_name] = std::move(ranks);
  };

  void record_accelerator_version(const std::string nccl_version) {
    if (!enabled_) {
      return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    nccl_version_ = std::move(nccl_version);
  };

  void update_state(Entry& r) {
    if (r.start_ != nullptr) {
      bool started = r.start_->query();
      if (started && !r.time_discovered_started_) {
        r.time_discovered_started_ = c10::getTime();
      }
    }
    if (r.end_ != nullptr) {
      bool completed = r.end_->query();
      if (completed && !r.time_discovered_completed_) {
        r.time_discovered_completed_ = c10::getTime();
      }
    }
  };

  std::vector<Entry> dump_entries() {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<Entry> result;
    result.reserve(entries_.size());
    result.insert(
        result.end(),
        entries_.begin() + static_cast<std::ptrdiff_t>(next_),
        entries_.end());
    result.insert(
        result.end(),
        entries_.begin(),
        entries_.begin() + static_cast<std::ptrdiff_t>(next_));
    // query any remaining events
    for (auto& r : result) {
      update_state(r);
      r.start_ = r.end_ = nullptr;
    }
    return result;
  };

  // Returns the entry with the given id, if it exists. Otherwise, returns
  // std::nullopt.
  std::optional<Entry> getEntry(std::optional<size_t> id) {
    if (!enabled_ || !id) {
      return std::nullopt;
    }

    std::unique_lock<std::mutex> guard(mutex_);
    Entry entry = entries_.at(*id % max_entries_);
    if (entry.id_ == *id) {
      return entry;
    } else {
      return std::nullopt;
    }
  };

  /*
  Mark an Event as completed and free its events.
  This is called by the watchdog thread, and is asynchronous from the
  perspective of the main thread.
  compute_duration defaults to true since retire_id is only called in the
  watchdog thread, which is currently a place we call cuda APIs which may hang,
  but care should be taken to avoid computing duration in any function that must
  never hang. (timing must also be enabled for compute_duration - see
  TORCH_NCCL_ENABLE_TIMING).
  */
  void retire_id(std::optional<size_t> id, bool compute_duration = true) {
    if (!enabled_ || !id) {
      return;
    };
    bool can_compute_duration = false;
    EventType* startEvent = nullptr;
    EventType* endEvent = nullptr;
    std::optional<float> duration = std::nullopt;

    std::unique_lock<std::mutex> guard(mutex_);

    Entry* entry = &entries_.at(*id % max_entries_);
    if (entry->id_ == *id) {
      update_state(*entry);

      if (compute_duration) {
        can_compute_duration = entry->time_discovered_completed_.has_value() &&
            entry->start_ && entry->end_;
        startEvent = entry->start_;
        endEvent = entry->end_;
      }
      entry->retired_ = true;
      entry->start_ = entry->end_ = nullptr;
    }

    if (can_compute_duration) {
      // Compute duration without without holding the lock, because
      // cudaEventDuration() can hang, and we need to acquire the lock before we
      // can dump(), which we never want to block.
      guard.unlock();
      duration = getDurationFromEvent(*startEvent, *endEvent);
      guard.lock();

      // Refresh the entry pointer, see if the entry has been overwritten
      entry = &entries_.at(*id % max_entries_);
      if (entry->id_ != *id) {
        LOG(INFO)
            << "retire_id abandoned for id " << *id
            << ", event was overwritten while waiting to compute duration.";
        return;
      }
      if (duration.has_value()) {
        entry->duration_ = duration;
      }
    }
  };

  const c10::List<c10::IValue> getCollectiveTrace(
      bool includeStacktraces,
      bool onlyActive) {
    auto entries = new_list();
    // Entries are returned in the order they were recorded
    auto result = dump_entries();
    std::vector<torch::CapturedTraceback*> tracebacks;
    torch::SymbolizedTracebacks stracebacks;
    std::vector<c10::IValue> all_frames;
    if (includeStacktraces) {
      for (auto& e : result) {
        tracebacks.push_back(e.traceback_.get());
      }
      stracebacks = torch::symbolize(tracebacks);
      for (const auto& f : stracebacks.all_frames) {
        auto d = new_dict();
        d.insert(name_key, f.funcname);
        d.insert(filename_key, f.filename);
        d.insert(line_key, int64_t(f.lineno));
        all_frames.emplace_back(std::move(d));
      }
    }
    for (auto i : c10::irange(result.size())) {
      auto dict = new_dict();
      auto& e = result.at(i);
      // Skip completed events
      if (onlyActive && e.time_discovered_completed_.has_value()) {
        continue;
      }
      if (includeStacktraces) {
        auto& tb = stracebacks.tracebacks.at(i);
        auto frames = new_list();
        for (auto frame : tb) {
          frames.push_back(all_frames.at(frame));
        }
        dict.insert(frames_key, frames);
      }

      dict.insert(record_id_key, int64_t(e.id_));
      dict.insert(pg_id_key, int64_t(e.pg_id_));
      dict.insert(pg_name_key, e.pg_name_);
      dict.insert(collective_seq_id_key, int64_t(e.collective_seq_id_));
      dict.insert(p2p_seq_id_key, int64_t(e.p2p_seq_id_));
      dict.insert(op_id_key, int64_t(e.op_id_));
      dict.insert(profiling_name_key, e.profiling_name_);
      dict.insert(time_created_key, int64_t(e.time_created_));
      if (e.duration_) {
        dict.insert(duration_key, *e.duration_);
      }

      auto it = e.sizes_.begin();
      auto read_sizes = [&](const c10::SmallVector<int64_t, 4>& dims) {
        auto sizes = new_list();
        for (auto dim : dims) {
          auto arg_sizes = new_list();
          for ([[maybe_unused]] auto i : c10::irange(dim)) {
            arg_sizes.push_back(*it++);
          }
          sizes.push_back(arg_sizes);
        }
        return sizes;
      };

      dict.insert(input_sizes_key, read_sizes(e.input_dims_));
      std::vector<std::string> input_dtypes_strs;
      input_dtypes_strs.reserve(e.input_dtypes_.size());
      for (const auto& input_dtype : e.input_dtypes_) {
        input_dtypes_strs.emplace_back(c10::toString(input_dtype));
      }
      dict.insert(input_dtypes_key, input_dtypes_strs);
      dict.insert(output_sizes_key, read_sizes(e.output_dims_));
      std::vector<std::string> output_dtypes_strs;
      output_dtypes_strs.reserve(e.output_dtypes_.size());
      for (const auto& output_dtype : e.output_dtypes_) {
        output_dtypes_strs.emplace_back(c10::toString(output_dtype));
      }
      dict.insert(output_dtypes_key, output_dtypes_strs);
      if (e.time_discovered_completed_.has_value()) {
        dict.insert(state_key, completed_state);
      } else if (e.time_discovered_started_.has_value()) {
        dict.insert(state_key, started_state);
      } else {
        dict.insert(state_key, scheduled_state);
      }

      dict.insert(
          time_discovered_started_key,
          e.time_discovered_started_.has_value()
              ? int64_t(*e.time_discovered_started_)
              : c10::IValue());
      dict.insert(
          time_discovered_completed_key,
          e.time_discovered_completed_.has_value()
              ? int64_t(*e.time_discovered_completed_)
              : c10::IValue());
      dict.insert(retired_key, e.retired_);
      dict.insert(timeout_key, e.timeout_ms_);
      dict.insert(is_p2p_key, e.isP2P_);

      entries.push_back(dict);
    }
    return entries;
  };

  // dump pg_entries
  const c10::Dict<c10::IValue, c10::IValue> getPgConfig() {
    auto pg_config = new_dict();
    for (const auto& [pg_name, ranks] : pg_name_to_ranks_) {
      auto pg_info = new_dict();
      pg_info.insert("name", std::get<0>(pg_name));
      pg_info.insert("desc", std::get<1>(pg_name));
      pg_info.insert("ranks", ranks_str(ranks));
      pg_config.insert(std::get<0>(pg_name), pg_info);
    }
    return pg_config;
  };

  const std::map<std::string, std::map<std::string, std::string>>
  getPgConfigJson() {
    std::map<std::string, std::map<std::string, std::string>> result;
    for (const auto& [pg_name, ranks] : pg_name_to_ranks_) {
      auto pg_info = std::map<std::string, std::string>();
      pg_info["name"] = std::get<0>(pg_name);
      pg_info["desc"] = std::get<1>(pg_name);
      pg_info["ranks"] = ranks_str(ranks);
      result.emplace(std::get<0>(pg_name), pg_info);
    }
    return result;
  };

  // dump pg_status
  const c10::Dict<c10::IValue, c10::IValue> getPgStatus() {
    auto all_pg_status = new_dict();
    for (const auto& [pg_id, status] : all_pg_status_) {
      auto pg_status = new_dict();
      pg_status.insert("last_enqueued_collective", status->lastEnqueuedSeq);
      pg_status.insert("last_started_collective", status->lastStartedSeq);
      pg_status.insert("last_completed_collective", status->lastCompletedSeq);
      all_pg_status.insert(std::to_string(pg_id), pg_status);
    }
    return all_pg_status;
  };

  const std::map<std::string, std::map<std::string, std::string>>
  getPgStatusJson() {
    std::map<std::string, std::map<std::string, std::string>> result;
    for (const auto& [pg_id, status] : all_pg_status_) {
      auto pg_status = std::map<std::string, std::string>();
      pg_status["last_enqueued_collective"] =
          std::to_string(status->lastEnqueuedSeq);
      pg_status["last_started_collective"] =
          std::to_string(status->lastStartedSeq);
      pg_status["last_completed_collective"] =
          std::to_string(status->lastCompletedSeq);
      result[std::to_string(pg_id)] = pg_status;
    }
    return result;
  };

  using json = nlohmann::json;
  json get_dump_json(bool includeCollectives, bool onlyActive) {
    json result;
    result[version_key_str] = version_val_str;
    result[nccl_version_key_str] = nccl_version_;
    result[pg_config_key_str] = getPgConfigJson();
    result[pg_status_key_str] = getPgStatusJson();
    // collective trace
    if (includeCollectives) {
      std::list<json> entries;
      for (auto& e : dump_entries()) {
        json j;
        if (onlyActive && e.time_discovered_completed_.has_value()) {
          continue;
        }
        j[record_id_key_str] = int64_t(e.id_);
        j[pg_id_key_str] = int64_t(e.pg_id_);
        j[pg_name_key_str] = e.pg_name_;
        j[collective_seq_id_key_str] = int64_t(e.collective_seq_id_);
        j[p2p_seq_id_key_str] = int64_t(e.p2p_seq_id_);
        j[op_id_key_str] = int64_t(e.op_id_);
        j[profiling_name_key_str] = e.profiling_name_;
        j[time_created_key_str] = int64_t(e.time_created_);
        if (e.duration_) {
          j[duration_key_str] = *e.duration_;
        }
        auto it = e.sizes_.begin();
        auto read_sizes = [&](const c10::SmallVector<int64_t, 4>& dims) {
          auto sizes = std::list<std::list<int64_t>>();
          for (auto dim : dims) {
            auto arg_sizes = std::list<int64_t>();
            for (auto i : c10::irange(dim)) {
              (void)i;
              arg_sizes.push_back(*it++);
            }
            sizes.push_back(arg_sizes);
          }
          return sizes;
        };
        j[input_sizes_key_str] = read_sizes(e.input_dims_);
        std::vector<std::string> input_dtypes_strs;
        input_dtypes_strs.reserve(e.input_dtypes_.size());
        for (const auto& input_dtype : e.input_dtypes_) {
          input_dtypes_strs.emplace_back(c10::toString(input_dtype));
        }
        j[input_dtypes_key_str] = input_dtypes_strs;
        j[output_sizes_key_str] = read_sizes(e.output_dims_);
        std::vector<std::string> output_dtypes_strs;
        output_dtypes_strs.reserve(e.output_dtypes_.size());
        for (const auto& output_dtype : e.output_dtypes_) {
          output_dtypes_strs.emplace_back(c10::toString(output_dtype));
        }
        j[output_dtypes_key_str] = output_dtypes_strs;
        if (e.time_discovered_completed_.has_value()) {
          j[state_key_str] = completed_state_str;
        } else if (e.time_discovered_started_.has_value()) {
          j[state_key_str] = started_state_str;
        } else {
          j[state_key_str] = scheduled_state_str;
        }
        j[time_discovered_started_key_str] =
            e.time_discovered_started_.has_value()
            ? int64_t(*e.time_discovered_started_)
            : 0;
        j[time_discovered_completed_key_str] =
            e.time_discovered_completed_.has_value()
            ? int64_t(*e.time_discovered_completed_)
            : 0;
        j[retired_key_str] = e.retired_;
        j[timeout_key_str] = e.timeout_ms_;
        j[is_p2p_key_str] = e.isP2P_;
        entries.emplace_back(j);
      }
      if (!entries.empty()) {
        result[entries_key_str] = entries;
      }
    }
    return result;
  };

  std::string dump_json(bool includeCollectives, bool onlyActive) {
    return (get_dump_json(includeCollectives, onlyActive)).dump();
  };

  std::string dump_json(
      const std::optional<std::unordered_map<
          std::string,
          std::unordered_map<std::string, std::string>>>& ncclDumpMap,
      bool includeCollectives,
      bool onlyActive) {
    // Adding ncclDumpMap to the json object is NCCL specific.
    auto result = get_dump_json(includeCollectives, onlyActive);
    if (ncclDumpMap.has_value()) {
      result[nccl_comm_key_str] = ncclDumpMap.value();
    }
    return result.dump();
  };

  c10::Dict<c10::IValue, c10::IValue> get_dump(
      bool includeCollectives,
      bool includeStackTraces,
      bool onlyActive) {
    STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.FlightRecorder__dump);
    auto result = new_dict();
    // common values
    result.insert(version_key, version_val);
    result.insert(pg_config_key, getPgConfig());
    result.insert(nccl_version_key_str, nccl_version_);
    result.insert(pg_status_key, getPgStatus());

    // collective trace
    if (includeCollectives) {
      result.insert(
          entries_key, getCollectiveTrace(includeStackTraces, onlyActive));
    }

    return result;
  };

  // dump all collectives + ncclDumpMap
  std::string dump(
      bool includeCollectives,
      bool includeStackTraces,
      bool onlyActive) {
    return pickle_str(
        get_dump(includeCollectives, includeStackTraces, onlyActive));
  };

  std::string dump(
      const std::optional<std::unordered_map<
          std::string,
          std::unordered_map<std::string, std::string>>>& ncclDumpMap,
      bool includeCollectives,
      bool includeStackTraces,
      bool onlyActive) {
    auto result = get_dump(includeCollectives, includeStackTraces, onlyActive);

    // convert ncclDumpMap into a dictionary
    auto per_comm_dict = new_dict();
    if (ncclDumpMap.has_value()) {
      for (const auto& [ncclId, ncclDump] : ncclDumpMap.value()) {
        auto inner_dict = new_dict();
        for (const auto& [key, value] : ncclDump) {
          inner_dict.insert(key, value);
        }
        per_comm_dict.insert(ncclId, inner_dict);
      }
    }
    if (!per_comm_dict.empty()) {
      result.insert(nccl_comm_key, per_comm_dict);
    }
    return pickle_str(result);
  };
};
} // namespace c10d
