// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/hooks/common/OpNameHelper.hpp>
#include <torch/csrc/comms/hooks/fr/FlightRecorder.hpp>

#include <c10/util/ApproximateClock.h>
#include <c10/util/WaitCounter.h>
#include <c10/util/irange.h>
#include <c10/util/thread_name.h>
#include <nlohmann/json.hpp>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>
#include <torch/csrc/jit/serialization/pickler.h>
#include <sstream>

namespace torch {
namespace comms {
namespace fr {

using json = nlohmann::json;

namespace {
std::string ranks_str(const std::vector<uint64_t>& ranks) {
  std::string str;
  for (size_t i = 0; i < ranks.size(); ++i) {
    if (i > 0) {
      str += ",";
    }
    str += std::to_string(ranks[i]);
  }
  return "[" + str + "]";
}

inline c10::Dict<c10::IValue, c10::IValue> new_dict() {
  return c10::Dict<c10::IValue, c10::IValue>(
      c10::AnyType::get(), c10::AnyType::get());
}

inline c10::List<c10::IValue> new_list() {
  return c10::List<c10::IValue>(c10::AnyType::get());
}

inline std::string pickle_str(const c10::IValue& v) {
  std::vector<char> result;
  {
    auto writer = [&](const char* data, size_t size) {
      result.insert(result.end(), data, data + size);
    };
    torch::jit::Pickler pickler(
        writer, nullptr, nullptr, nullptr, nullptr, false);
    pickler.protocol();
    pickler.pushIValue(v);
    pickler.stop();
  }
  return std::string(result.begin(), result.end());
}
} // namespace

namespace {

// Static registration of the torch_comms_fr_trace_json handler. The name is
// distinct from the standalone torchcomms package's "torchcomms_fr_trace_json"
// handler so the two can coexist in one process (e.g. when both are installed
// in CI) without a duplicate-registration abort.
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
c10d::control_plane::RegisterHandler torchcommsFrTraceJsonRegistration(
    "torch_comms_fr_trace_json",
    [](const c10d::control_plane::Request& req,
       c10d::control_plane::Response& res) {
      const auto& params = req.params();
      size_t validParamCount = 0;

      // valid params
      const std::string includeCollectivesStr = "includecollectives";
      const std::string onlyActiveStr = "onlyactive";

      std::unordered_map<std::string, bool> processedParams = {
          {includeCollectivesStr, true}, {onlyActiveStr, false}};

      for (const auto& [paramName, paramValue] : params) {
        auto it = processedParams.find(paramName);
        if (it != processedParams.end()) {
          validParamCount++;
          if (paramValue == "true") {
            it->second = true;
          } else if (paramValue == "false") {
            it->second = false;
          } else {
            res.setStatus(400);
            res.setContent(
                "Invalid value for " + paramName +
                    " valid values are true or false",
                "text/plain");
            return;
          }
        }
      }
      if (validParamCount < params.size()) {
        res.setStatus(400);
        res.setContent(
            "Invalid parameters - unexpected param passed in", "text/plain");
        return;
      }

      auto* recorder = FlightRecorder::get();
      auto trace = recorder->dump_json(
          std::nullopt,
          processedParams[includeCollectivesStr],
          processedParams[onlyActiveStr]);
      res.setContent(std::move(trace), "application/json");
      res.setStatus(200);
    });

} // namespace
float getDurationFromEvent(c10::Event& start, c10::Event& end);

// Returns the traceback of current entry, in string form.
// Note: `getTraceback` invokes `torch::symbolize`, which may need to acquire
// the GIL. If you don't want to block the current thread or take the risk of a
// GIL deadlock, you can use an asynchronous calling mechanism like std::async.
std::string FlightRecorder::Entry::getTraceback() {
  torch::CapturedTraceback* traceback = traceback_.get();
  torch::SymbolizedTracebacks s_tbs = torch::symbolize({traceback});
  // We use 0 because we only have one traceback here.
  const auto& s_tb = s_tbs.tracebacks.at(0);
  std::stringstream oss;
  for (auto idx : c10::irange(s_tb.size())) {
    auto frame_id = s_tb[idx];
    const auto& frame = s_tbs.all_frames.at(frame_id);
    oss << '#' << idx << ' ' << frame.funcname << " from " << frame.filename
        << ':' << frame.lineno << '\n';
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

void FlightRecorder::record(
    size_t pg_id,
    const std::tuple<std::string, std::string>& pg_name,
    size_t op_id,
    std::string profiling_name,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs,
    c10::Event* start,
    c10::Event* end,
    std::chrono::milliseconds timeout_ms,
    std::shared_ptr<ProcessGroupStatus> pg_status) {
  if (!enabled_) {
    return;
  }

  auto traceback =
      torch::CapturedTraceback::gather(true, true, capture_cpp_stack_);

  std::lock_guard<std::mutex> guard(mutex_);

  if (all_pg_status_.find(pg_id) == all_pg_status_.end()) {
    // Current pg_status is not in FR.
    all_pg_status_[pg_id] = std::move(pg_status);
  }

  TORCH_CHECK(
      reset_epoch_start_idx_.find(reset_epoch_) !=
      reset_epoch_start_idx_.end());

  size_t collective_seq_id = collective_seq_id_++;

  auto te = Entry{
      id_,
      reset_epoch_,
      pg_id,
      pg_name,
      collective_seq_id,
      op_id,
      std::move(profiling_name),
      std::move(traceback),
      start,
      end,
      c10::getTime(),
      timeout_ms.count(),
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

  const auto idx = op_id % max_entries_;
  latest_op_id_ = op_id;

  if (entries_.size() < max_entries_) {
    entries_.emplace_back(std::move(te));
    TORCH_CHECK(entries_.size() == idx + 1);
  } else {
    entries_[idx] = std::move(te);
  }

  // Store the mapping from op_id to (id_, reset_epoch_) for retire_id lookup
  op_id_to_id_and_epoch_[op_id] = std::make_pair(id_, reset_epoch_);

  id_++;
}

void FlightRecorder::record_pg_ranks(
    const std::tuple<std::string, std::string>& pg_name,
    std::vector<uint64_t> ranks) {
  if (!enabled_) {
    return;
  }
  std::lock_guard<std::mutex> guard(mutex_);
  pg_name_to_ranks_[pg_name] = std::move(ranks);
}

void FlightRecorder::record_accelerator_version(std::string comm_lib_version) {
  if (!enabled_) {
    return;
  }
  std::lock_guard<std::mutex> guard(mutex_);
  comm_lib_version_ = std::move(comm_lib_version);
}

void FlightRecorder::update_state(Entry& r) {
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

std::vector<FlightRecorder::Entry> FlightRecorder::dump_entries() {
  std::vector<Entry> result;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    // Return empty if no entries or max_entries_ is 0
    if (max_entries_ == 0 || entries_.empty()) {
      return result;
    }
    // Filter entries during insertion - only keep entries from current epoch
    auto filter = [this](const Entry& e) {
      return e.reset_epoch_ == reset_epoch_;
    };
    const auto next = ((latest_op_id_ + 1) % max_entries_);
    std::copy_if(
        entries_.begin() + static_cast<std::ptrdiff_t>(next),
        entries_.end(),
        std::back_inserter(result),
        filter);
    std::copy_if(
        entries_.begin(),
        entries_.begin() + static_cast<std::ptrdiff_t>(next),
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

// Returns the index in entries_ for the given id and reset_epoch.
// Caller must hold mutex_lock before calling this method.
size_t FlightRecorder::getIdxFromId(size_t id, size_t reset_epoch) const {
  // Look up the starting idx for the given reset epoch
  auto it = reset_epoch_start_idx_.find(reset_epoch);
  TORCH_CHECK(it != reset_epoch_start_idx_.end());
  // Calculate idx based on where the epoch started
  return (it->second + id) % max_entries_;
}

// Returns the entry with the given id and reset_epoch, if it exists. Otherwise,
// returns std::nullopt.
std::optional<FlightRecorder::Entry> FlightRecorder::getEntry(
    std::optional<size_t> id,
    std::optional<size_t> reset_epoch) {
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

void FlightRecorder::retire_id(
    std::optional<size_t> id,
    bool compute_duration) {
  if (!enabled_ || !id) {
    return;
  }

  bool can_compute_duration = false;
  c10::Event* startEvent = nullptr;
  c10::Event* endEvent = nullptr;
  std::optional<float> duration = std::nullopt;

  std::unique_lock<std::mutex> guard(mutex_);

  // Look up the (id_, reset_epoch_) pair for this op_id
  auto it = op_id_to_id_and_epoch_.find(*id);
  if (it == op_id_to_id_and_epoch_.end()) {
    return; // op_id not found
  }
  auto [internal_id, reset_epoch] = it->second;
  op_id_to_id_and_epoch_.erase(it);

  Entry* entry = &entries_.at(getIdxFromId(internal_id, reset_epoch));
  if (entry->id_ == internal_id && entry->reset_epoch_ == reset_epoch) {
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
    entry = &entries_.at(getIdxFromId(*id, reset_epoch));
    if (!(entry->id_ == *id && entry->reset_epoch_ == reset_epoch)) {
      LOG(INFO) << "retire_id abandoned for id " << *id
                << ", event was overwritten while waiting to compute duration.";
      return;
    }
    if (duration.has_value()) {
      entry->duration_ = duration;
    }
  }
}

void FlightRecorder::reset_all() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (!entries_.empty()) {
    // Soft delete: increment epoch to mark all existing entries as old
    // Store where the new epoch starts in the circular buffer
    reset_epoch_++;
    reset_epoch_start_idx_[reset_epoch_] = (latest_op_id_ + 1) % max_entries_;
    id_ = 0;
  }
}

const c10::List<c10::IValue> FlightRecorder::getCollectiveTrace(
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
    // TODO: Original FR had e.time_discovered_completed_.has_value()
    // instead of e.retired_
    if (onlyActive && e.retired_) {
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
    dict.insert(p2p_seq_id_key, int64_t(0));
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
        for ([[maybe_unused]] auto j : c10::irange(dim)) {
          arg_sizes.push_back(*it++);
        }
        sizes.push_back(arg_sizes);
      }
      return sizes; // NOLINT(clang-diagnostic-nrvo)
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
    dict.insert(is_p2p_key, false);

    entries.push_back(dict);
  }
  return entries;
}

const c10::Dict<c10::IValue, c10::IValue> FlightRecorder::getPgConfig() {
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

const std::map<std::string, std::map<std::string, std::string>> FlightRecorder::
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
}

const c10::Dict<c10::IValue, c10::IValue> FlightRecorder::getPgStatus() {
  auto all_pg_status = new_dict();
  for (const auto& [pg_id, status] : all_pg_status_) {
    if (status == nullptr) {
      continue;
    }
    auto pg_status = new_dict();
    pg_status.insert("last_enqueued_collective", status->lastEnqueuedSeq);
    pg_status.insert("last_started_collective", status->lastStartedSeq);
    pg_status.insert("last_completed_collective", status->lastCompletedSeq);
    all_pg_status.insert(std::to_string(pg_id), pg_status);
  }
  return all_pg_status;
}

const std::map<std::string, std::map<std::string, std::string>> FlightRecorder::
    getPgStatusJson() {
  std::map<std::string, std::map<std::string, std::string>> result;
  for (const auto& [pg_id, status] : all_pg_status_) {
    if (status == nullptr) {
      continue;
    }
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
std::string FlightRecorder::dump_json(
    const std::optional<std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::string>>>& extraDumpMap,
    bool includeCollectives,
    bool onlyActive) {
  json result;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    result[version_key_str] = version_val_str;
    result[comm_lib_version_key_str] = comm_lib_version_;
    result[pg_config_key_str] = getPgConfigJson();
    result[pg_status_key_str] = getPgStatusJson();
  }

  // collective trace
  if (includeCollectives) {
    std::list<json> entries;
    for (auto& e : dump_entries()) {
      json j;
      if (onlyActive && e.retired_) {
        continue;
      }
      j[record_id_key_str] = int64_t(e.id_);
      j[pg_id_key_str] = int64_t(e.pg_id_);
      j[pg_name_key_str] = e.pg_name_;
      j[thread_name_key_str] = e.thread_name_;
      j[thread_id_key_str] = c10::str(e.thread_id_);
      j[collective_seq_id_key_str] = int64_t(e.collective_seq_id_);
      j[p2p_seq_id_key_str] = int64_t(0);
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
        return sizes; // NOLINT(clang-diagnostic-nrvo)
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
      j[is_p2p_key_str] = false;
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

std::string FlightRecorder::dump(
    const std::optional<std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::string>>>& extraDumpMap,
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.FlightRecorder__dump);
  auto result = new_dict();
  // common values
  {
    std::lock_guard<std::mutex> guard(mutex_);
    result.insert(version_key, version_val);
    result.insert(pg_config_key, getPgConfig());
    result.insert(comm_lib_version_key_str, comm_lib_version_);
    result.insert(pg_status_key, getPgStatus());
  }

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

size_t FlightRecorder::size() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return id_;
}

void DebugInfoWriter::write(const std::string& trace) {
  std::string filename = filename_;
  if (enable_dynamic_filename_) {
    LOG(INFO) << "Writing Flight Recorder debug info to a dynamic file name";
    std::string defaultFileName;
    filename =
        c10::str(env_to_value("TORCHCOMM_FR_DUMP_TEMP_FILE", defaultFileName));
  } else {
    LOG(INFO) << "Writing Flight Recorder debug info to a static file name";
  }
  // Open a file for writing. The ios::binary flag is used to write data as
  // binary.
  std::ofstream file(filename, std::ios::binary);

  // Check if the file was opened successfully.
  if (!file.is_open()) {
    LOG(ERROR) << "Error opening file for writing Flight Recorder debug info: "
               << filename;
    return;
  }

  if (!file.write(trace.data(), static_cast<std::streamsize>(trace.size()))) {
    const auto bad = file.bad();
    LOG(ERROR) << "Error writing Flight Recorder debug info to file: "
               << filename << " bad bit: " << bad;
    return;
  }

  // Flush the buffer to ensure data is written to the file
  file.flush();
  if (file.bad()) {
    LOG(ERROR) << "Error flushing Flight Recorder debug info: " << filename;
    return;
  }

  LOG(INFO) << "Finished writing Flight Recorder debug info to " << filename;
}

DebugInfoWriter& DebugInfoWriter::getWriter(int rank) {
  if (writer_ == nullptr) {
// Attempt to write to running user's HOME directory cache folder - if it
// exists.
#ifdef _WIN32
    const char* cacheHome = nullptr;
#else
    // Uses XDG_CACHE_HOME if it's set
    const char* cacheHome = std::getenv("XDG_CACHE_HOME");
#endif
    std::string cacheRoot;
    if (cacheHome) {
      cacheRoot = cacheHome;
    } else {
      std::string defaultCacheRoot = "/tmp";
      cacheRoot = env_to_value("HOME", defaultCacheRoot) + "/.cache";
    }
    auto cacheDirPath = std::filesystem::path(cacheRoot + "/torch");
    // Create the .cache directory if it doesn't exist
    c10::filesystem::create_directories(cacheDirPath);
    auto defaultLocation = cacheDirPath / "comm_lib_trace_rank_";

    // For internal bc compatibility, we keep the old the ENV check.
    std::string fileNamePrefix =
        env_to_value("TORCHCOMM_FR_DUMP_TEMP_FILE", defaultLocation.string());
    bool useDynamicFileName =
        env_to_value("TORCHCOMM_FR_DUMP_DYNAMIC_FILE_NAME", false);
    // Using std::unique_ptr here to auto-delete the writer object
    // when the pointer itself is destroyed.
    std::unique_ptr<DebugInfoWriter> writerPtr(
        new DebugInfoWriter(fileNamePrefix, rank, useDynamicFileName));
    DebugInfoWriter::registerWriter(std::move(writerPtr));
  }
  return *writer_;
}

void DebugInfoWriter::registerWriter(std::unique_ptr<DebugInfoWriter> writer) {
  if (hasWriterRegistered_.load()) {
    TORCH_WARN_ONCE(
        "DebugInfoWriter has already been registered, and since we need the writer to stay "
        "outside ProcessGroup, user needs to ensure that this extra registration is indeed needed. "
        "And we will only use the last registered writer.");
  }
  hasWriterRegistered_.store(true);
  writer_ = std::move(writer);
}

std::unique_ptr<DebugInfoWriter> DebugInfoWriter::writer_ = nullptr;
std::atomic<bool> DebugInfoWriter::hasWriterRegistered_(false);

float getDurationFromEvent(
    [[maybe_unused]] c10::Event& startEvent,
    [[maybe_unused]] c10::Event& endEvent) {
  TORCH_CHECK(false, "getDuration not supported by c10::Event.");
}

// ============================================================================
// TorchComm hooks
// ============================================================================

FlightRecorderHook::FlightRecorderHook(size_t max_entries, bool isolated) {
  if (isolated) {
    // Reset global op_id generator so this isolated instance gets
    // op_ids starting from 0. This ensures tests don't share op_ids.
    ::torch::comms::resetGlobalOpIdGenerator();
    recorder_ = new FlightRecorder(max_entries, true);
    owns_recorder_ = true;
  } else {
    recorder_ = FlightRecorder::get();
    owns_recorder_ = false;
  }
}

FlightRecorderHook::~FlightRecorderHook() {
  if (owns_recorder_) {
    delete recorder_;
  }
}

void FlightRecorderHook::registerWithComm(std::shared_ptr<TorchComm> comm) {
  std::string comm_name(comm->getCommName());
  // Use registration count as pg_id (unique ID for each registered
  // communicator)
  size_t pg_id = registrations_.size();
  // Use backend name as description for the process group
  std::string pg_desc(comm->getBackend());

  auto pgName = std::make_tuple(comm_name, pg_desc);
  // Get ranks from the communicator - for split comms this will be the
  // global ranks from the parent
  auto comm_ranks = comm->getRanks();
  std::vector<uint64_t> pg_ranks;
  pg_ranks.reserve(comm_ranks.size());
  for (int rank : comm_ranks) {
    pg_ranks.push_back(static_cast<uint64_t>(rank));
  }
  recorder_->record_pg_ranks(pgName, pg_ranks);

  auto self = shared_from_this();

  // Register pre-hook - records the operation
  auto pre_hook_handle = comm->registerPreHook(
      [self, comm_name, pg_id, pg_desc](size_t op_id, const PreHookArgs& args) {
        self->onPreHook(comm_name, pg_id, pg_desc, op_id, args);
      });

  // Register post-hook - called via work callback when work completes
  // The post-hook is invoked by TorchComm when the work's callback fires
  auto post_hook_handle =
      comm->registerPostHook([self](size_t op_id, const PostHookArgs& args) {
        self->onPostHook(op_id, args);
      });

  // Register abort hook - called before aborting to dump flight recorder data
  int rank = comm->getRank();
  comm->registerAbortHook([self, rank]() { self->dump_file(rank); });

  registrations_.emplace_back(comm, pg_id, pg_desc);
  enabled_ = true;
}

void FlightRecorderHook::onPreHook(
    const std::string& comm_name,
    size_t pg_id,
    const std::string& pg_desc,
    size_t op_id,
    const PreHookArgs& args) {
  auto name = getOpName(args);
  if (!enabled_) {
    return;
  }

  if (name == OpName::finalize) {
    return;
  }

  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> outputs;

  // Extract input/output tensors from the per-collective variant args
  std::visit(
      [&inputs, &outputs](const auto& a) {
        using T = std::decay_t<decltype(a)>;
        if constexpr (
            std::is_same_v<T, SendPreHookArgs> ||
            std::is_same_v<T, BroadcastPreHookArgs> ||
            std::is_same_v<T, AllReducePreHookArgs> ||
            std::is_same_v<T, ReducePreHookArgs>) {
          inputs.push_back(a.tensor);
          outputs.push_back(a.tensor);
        } else if constexpr (std::is_same_v<T, RecvPreHookArgs>) {
          outputs.push_back(a.tensor);
        } else if constexpr (
            std::is_same_v<T, AllGatherSinglePreHookArgs> ||
            std::is_same_v<T, ReduceScatterSinglePreHookArgs> ||
            std::is_same_v<T, AllToAllSinglePreHookArgs> ||
            std::is_same_v<T, AllToAllVSinglePreHookArgs> ||
            std::is_same_v<T, GatherSinglePreHookArgs>) {
          inputs.push_back(a.input);
          outputs.push_back(a.output);
        } else if constexpr (
            std::is_same_v<T, AllGatherPreHookArgs> ||
            std::is_same_v<T, AllGatherVPreHookArgs>) {
          inputs.push_back(a.input);
          outputs.insert(outputs.end(), a.output.begin(), a.output.end());
        } else if constexpr (
            std::is_same_v<T, ReduceScatterPreHookArgs> ||
            std::is_same_v<T, ReduceScatterVPreHookArgs>) {
          inputs.insert(inputs.end(), a.input.begin(), a.input.end());
          outputs.push_back(a.output);
        } else if constexpr (std::is_same_v<T, AllToAllPreHookArgs>) {
          inputs.insert(inputs.end(), a.input.begin(), a.input.end());
          outputs.insert(outputs.end(), a.output.begin(), a.output.end());
        } else if constexpr (std::is_same_v<T, ScatterPreHookArgs>) {
          inputs.insert(inputs.end(), a.input.begin(), a.input.end());
          outputs.push_back(a.output);
        } else if constexpr (std::is_same_v<T, GatherPreHookArgs>) {
          inputs.push_back(a.input);
          outputs.insert(outputs.end(), a.output.begin(), a.output.end());
        }
        // BarrierPreHookArgs, SplitPreHookArgs, NewWindowPreHookArgs,
        // FinalizePreHookArgs: no tensors
      },
      args);

  auto pg_name = std::make_tuple(comm_name, pg_desc);

  // Use "<backend>:<op>" format as expected by the FR trace analyzer
  std::string profiling_name =
      std::string(pg_desc) + ":" + std::string(opToString(name));

  // TODO: Create start/end events for accurate timing
  // For now, pass nullptr - timing will be based on CPU timestamps

  recorder_->record(
      pg_id,
      pg_name,
      op_id,
      std::move(profiling_name),
      inputs,
      outputs,
      nullptr, // start event
      nullptr, // end event
      std::chrono::milliseconds(600000), // 10 minute default timeout
      nullptr // pg_status
  );
}

void FlightRecorderHook::onPostHook(size_t op_id, const PostHookArgs& args) {
  if (!enabled_) {
    return;
  }
  if (std::get_if<FinalizePostHookArgs>(&args)) {
    return;
  }
  recorder_->retire_id(op_id, false);

  // Handle split operations - register the new communicator with flight
  // recorder
  if (auto* split = std::get_if<SplitPostHookArgs>(&args)) {
    if (auto new_comm = split->new_comm.lock()) {
      splitHook(new_comm);
    }
  }
}

void FlightRecorderHook::splitHook(std::shared_ptr<TorchComm> new_comm) {
  if (!enabled_ || !new_comm) {
    return;
  }
  registerWithComm(std::move(new_comm));
}

std::string FlightRecorderHook::dump_json(bool include_completed) const {
  return recorder_->dump_json(
      std::nullopt, // no extra dump map
      true, // include collectives
      !include_completed // onlyActive = !include_completed
  );
}

void FlightRecorderHook::dump_file(int rank, bool include_completed) const {
  LOG(INFO) << "Writing Flight Recorder debug info to file";
  std::string trace = recorder_->dump(
      std::unordered_map<
          std::string,
          std::unordered_map<std::string, std::string>>{},
      true, // includeCollectives
      false, // includeStackTraces
      !include_completed // onlyActive
  );
  DebugInfoWriter& writer = DebugInfoWriter::getWriter(rank);
  writer.write(trace);
}

void FlightRecorderHook::reset() {
  recorder_->reset_all();
}

bool FlightRecorderHook::isEnabled() const {
  return enabled_;
}

size_t FlightRecorderHook::size() const {
  return recorder_->size();
}

} // namespace fr
} // namespace comms
} // namespace torch
