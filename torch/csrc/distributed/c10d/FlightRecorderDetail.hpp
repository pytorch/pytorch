#include <nlohmann/json.hpp>

#include <c10/util/WaitCounter.h>
#include <c10/util/thread_name.h>

#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>

namespace c10d {

template <typename EventType>
float getDurationFromEvent(EventType& start, EventType& end);

// Returns the traceback of current entry, in string form.
// Note: `getTraceback` invokes `torch::symbolize`, which may need to acquire
// the GIL. If you don't want to block the current thread or take the risk of a
// GIL deadlock, you can use an asynchronous calling mechanism like std::async.
template <typename EventType>
std::string FlightRecorder<EventType>::Entry::getTraceback() {
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
}

template <typename EventType>
std::optional<size_t> FlightRecorder<EventType>::record(
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
  auto result = recordWithResetEnabled(
      pg_id,
      pg_name,
      collective_seq_id,
      p2p_seq_id,
      op_id,
      std::move(profiling_name),
      inputs,
      outputs,
      start,
      end,
      timeout_ms,
      std::move(pg_status),
      isP2P);
  return result.id;
}

template <typename EventType>
typename FlightRecorder<EventType>::TraceIdentifier FlightRecorder<EventType>::
    recordWithResetEnabled(
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
    return TraceIdentifier{std::nullopt, std::nullopt};
  }
  if (all_pg_status_.find(pg_id) == all_pg_status_.end()) {
    // Current pg_status is not in FR.
    all_pg_status_[pg_id] = std::move(pg_status);
  }
  auto traceback =
      torch::CapturedTraceback::gather(true, true, capture_cpp_stack_);
  std::lock_guard<std::mutex> guard(mutex_);

  TORCH_CHECK(
      reset_epoch_start_idx_.find(reset_epoch_) !=
      reset_epoch_start_idx_.end());

  auto te = Entry{
      id_,
      reset_epoch_,
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
      std::this_thread::get_id(),
      c10::getThreadName(),
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

  const auto next = next_++;

  if (entries_.size() < max_entries_) {
    entries_.emplace_back(std::move(te));
  } else {
    entries_[next] = std::move(te);
  }

  if (next_ == max_entries_) {
    next_ = 0;
  }

  const auto id = id_++;
  return TraceIdentifier{id, reset_epoch_};
}

template <typename EventType>
void FlightRecorder<EventType>::record_pg_ranks(
    const std::tuple<std::string, std::string>& pg_name,
    std::vector<uint64_t> ranks) {
  if (!enabled_) {
    return;
  }
  std::lock_guard<std::mutex> guard(mutex_);
  pg_name_to_ranks_[pg_name] = std::move(ranks);
}

template <typename EventType>
void FlightRecorder<EventType>::record_accelerator_version(
    const std::string comm_lib_version) {
  if (!enabled_) {
    return;
  }
  std::lock_guard<std::mutex> guard(mutex_);
  comm_lib_version_ = std::move(comm_lib_version);
}

template <typename EventType>
void FlightRecorder<EventType>::update_state(Entry& r) {
  try {
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
  } catch (std::exception& e) {
    LOG(ERROR) << "Failed to update state for entry " << r.id_ << ": "
               << r.profiling_name_ << " with error: " << e.what();
  }
}

template <typename EventType>
std::vector<typename FlightRecorder<EventType>::Entry> FlightRecorder<
    EventType>::dump_entries() {
  std::vector<Entry> result;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    // Filter entries during insertion - only keep entries from current epoch
    auto filter = [this](const Entry& e) {
      return e.reset_epoch_ == reset_epoch_;
    };
    std::copy_if(
        entries_.begin() + static_cast<std::ptrdiff_t>(next_),
        entries_.end(),
        std::back_inserter(result),
        filter);
    std::copy_if(
        entries_.begin(),
        entries_.begin() + static_cast<std::ptrdiff_t>(next_),
        std::back_inserter(result),
        filter);
  }
  // query any remaining events
  for (auto& r : result) {
    update_state(r);
    r.start_ = r.end_ = nullptr;
  }
  return result;
}

template <typename EventType>
// Returns the index in entries_ for the given id and reset_epoch.
// Caller must hold mutex_lock before calling this method.
size_t FlightRecorder<EventType>::getIdxFromId(size_t id, size_t reset_epoch)
    const {
  // Look up the starting idx for the given reset epoch
  auto it = reset_epoch_start_idx_.find(reset_epoch);
  TORCH_CHECK(it != reset_epoch_start_idx_.end());
  // Calculate idx based on where the epoch started
  return (it->second + id) % max_entries_;
}

template <typename EventType>
// Returns the entry with the given id and reset_epoch, if it exists. Otherwise,
// returns std::nullopt.
std::optional<typename FlightRecorder<EventType>::Entry> FlightRecorder<
    EventType>::
    getEntry(std::optional<size_t> id, std::optional<size_t> reset_epoch) {
  if (!enabled_ || !id || !reset_epoch) {
    return std::nullopt;
  }

  std::unique_lock<std::mutex> guard(mutex_);
  Entry entry = entries_.at(getIdxFromId(*id, *reset_epoch));
  if (entry.id_ == *id && entry.reset_epoch_ == *reset_epoch) {
    return entry;
  }
  return std::nullopt;
}

template <typename EventType>
std::optional<typename FlightRecorder<EventType>::Entry> FlightRecorder<
    EventType>::getEntry(std::optional<size_t> id) {
  return getEntry(id, 0);
}

template <typename EventType>
void FlightRecorder<EventType>::retire_id(
    std::optional<size_t> id,
    std::optional<size_t> reset_epoch,
    bool compute_duration) {
  if (!enabled_ || !id || !reset_epoch) {
    return;
  }

  bool can_compute_duration = false;
  EventType* startEvent = nullptr;
  EventType* endEvent = nullptr;
  std::optional<float> duration = std::nullopt;

  std::unique_lock<std::mutex> guard(mutex_);

  Entry* entry = &entries_.at(getIdxFromId(*id, *reset_epoch));
  if (entry->id_ == *id && entry->reset_epoch_ == *reset_epoch) {
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
    duration = getDurationFromEvent<EventType>(*startEvent, *endEvent);
    guard.lock();

    // Refresh the entry pointer, see if the entry has been overwritten
    entry = &entries_.at(getIdxFromId(*id, *reset_epoch));
    if (!(entry->id_ == *id && entry->reset_epoch_ == *reset_epoch)) {
      LOG(INFO) << "retire_id abandoned for id " << *id
                << ", event was overwritten while waiting to compute duration.";
      return;
    }
    if (duration.has_value()) {
      entry->duration_ = duration;
    }
  }
}

template <typename EventType>
void FlightRecorder<EventType>::retire_id(
    std::optional<size_t> id,
    bool compute_duration) {
  retire_id(id, 0, compute_duration);
}

template <typename EventType>
void FlightRecorder<EventType>::reset_all() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (!entries_.empty()) {
    // Soft delete: increment epoch to mark all existing entries as old
    // Store where the new epoch starts in the circular buffer
    reset_epoch_++;
    reset_epoch_start_idx_[reset_epoch_] = next_;
    id_ = 0;
  }
}

template <typename EventType>
const c10::List<c10::IValue> FlightRecorder<EventType>::getCollectiveTrace(
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
    dict.insert(thread_name_key, e.thread_name_);
    dict.insert(thread_id_key, c10::str(e.thread_id_));
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
}

template <typename EventType>
const c10::Dict<c10::IValue, c10::IValue> FlightRecorder<
    EventType>::getPgConfig() {
  auto pg_config = new_dict();
  for (const auto& [pg_name, ranks] : pg_name_to_ranks_) {
    auto pg_info = new_dict();
    pg_info.insert("name", std::get<0>(pg_name));
    pg_info.insert("desc", std::get<1>(pg_name));
    pg_info.insert("ranks", ranks_str(ranks));
    pg_config.insert(std::get<0>(pg_name), pg_info);
  }
  return pg_config;
}

template <typename EventType>
const std::map<std::string, std::map<std::string, std::string>> FlightRecorder<
    EventType>::getPgConfigJson() {
  std::map<std::string, std::map<std::string, std::string>> result;
  for (const auto& [pg_name, ranks] : pg_name_to_ranks_) {
    auto pg_info = std::map<std::string, std::string>();
    pg_info["name"] = std::get<0>(pg_name);
    pg_info["desc"] = std::get<1>(pg_name);
    pg_info["ranks"] = ranks_str(ranks);
    result.emplace(std::get<0>(pg_name), pg_info);
  }
  return result;
}

template <typename EventType>
const c10::Dict<c10::IValue, c10::IValue> FlightRecorder<
    EventType>::getPgStatus() {
  auto all_pg_status = new_dict();
  for (const auto& [pg_id, status] : all_pg_status_) {
    auto pg_status = new_dict();
    pg_status.insert("last_enqueued_collective", status->lastEnqueuedSeq);
    pg_status.insert("last_started_collective", status->lastStartedSeq);
    pg_status.insert("last_completed_collective", status->lastCompletedSeq);
    all_pg_status.insert(std::to_string(pg_id), pg_status);
  }
  return all_pg_status;
}

template <typename EventType>
const std::map<std::string, std::map<std::string, std::string>> FlightRecorder<
    EventType>::getPgStatusJson() {
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
}

using json = nlohmann::json;
template <typename EventType>
std::string FlightRecorder<EventType>::dump_json(
    const std::optional<std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::string>>>& extraDumpMap,
    bool includeCollectives,
    bool onlyActive) {
  json result;
  result[version_key_str] = version_val_str;
  result[comm_lib_version_key_str] = comm_lib_version_;
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
      j[thread_name_key_str] = e.thread_name_;
      j[thread_id_key_str] = c10::str(e.thread_id_);
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

  if (extraDumpMap.has_value()) {
    result[nccl_comm_key_str] = extraDumpMap.value();
  }
  return result.dump();
}

template <typename EventType>
std::string FlightRecorder<EventType>::dump(
    const std::optional<std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::string>>>& extraDumpMap,
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.FlightRecorder__dump);
  auto result = new_dict();
  // common values
  result.insert(version_key, version_val);
  result.insert(pg_config_key, getPgConfig());
  result.insert(comm_lib_version_key_str, comm_lib_version_);
  result.insert(pg_status_key, getPgStatus());

  // collective trace
  if (includeCollectives) {
    result.insert(
        entries_key, getCollectiveTrace(includeStackTraces, onlyActive));
  }

  // convert extraDumpMap into a dictionary
  auto per_comm_dict = new_dict();
  if (extraDumpMap.has_value()) {
    for (const auto& [ncclId, ncclDump] : extraDumpMap.value()) {
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
}
} // namespace c10d
