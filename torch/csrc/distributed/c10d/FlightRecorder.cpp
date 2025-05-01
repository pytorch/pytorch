#ifdef USE_C10D_NCCL
#include <cuda_runtime.h>
#endif // USE_C10D_NCCL
#include <nlohmann/json.hpp>
#ifndef _WIN32
#include <sys/stat.h>
#else
#include <direct.h>
#endif
#include <fstream>
#include <mutex>
#include <vector>

#include <c10/util/WaitCounter.h>

#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>
#ifdef USE_C10D_NCCL
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif // USE_C10D_NCCL
#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

namespace c10d {

template <typename T, typename EntryT, typename EventType>
std::optional<size_t> recordImpl(
    T& obj,
    EntryT&& te,
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
  if (!obj.enabled_) {
    return std::nullopt;
  }
  if (obj.all_pg_status_.find(pg_id) == obj.all_pg_status_.end()) {
    // Current pg_status is not in FR.
    obj.all_pg_status_[pg_id] = std::move(pg_status);
  }
  auto traceback =
      torch::CapturedTraceback::gather(true, true, obj.capture_cpp_stack_);
  std::lock_guard<std::mutex> guard(obj.mutex_);

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

  if (obj.entries_.size() < obj.max_entries_) {
    obj.entries_.emplace_back(std::move(te));
  } else {
    obj.entries_[obj.next_++] = std::move(te);
    if (obj.next_ == obj.max_entries_) {
      obj.next_ = 0;
    }
  }
  return obj.id_++;
}

template <typename EntryT>
void update_state_impl(EntryT& r) {
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
}

template <typename EntryT>
std::vector<EntryT> dump_entries_impl(
    const std::vector<EntryT>& entries,
    const size_t next,
    std::mutex& mutex) {
  std::lock_guard<std::mutex> guard(mutex);
  std::vector<EntryT> result;
  result.reserve(entries.size());
  result.insert(
      result.end(),
      entries.begin() + static_cast<std::ptrdiff_t>(next),
      entries.end());
  result.insert(
      result.end(),
      entries.begin(),
      entries.begin() + static_cast<std::ptrdiff_t>(next));
  // query any remaining events
  for (auto& r : result) {
    update_state_impl(r);
    r.start_ = r.end_ = nullptr;
  }
  return result;
}

template <typename EntryT>
std::optional<EntryT> getEntryImpl(
    const std::vector<EntryT>& entries,
    const bool enabled,
    std::mutex& mutex,
    const size_t max_entries,
    const std::optional<size_t> id) {
  if (!enabled || !id) {
    return std::nullopt;
  }

  std::unique_lock<std::mutex> guard(mutex);
  EntryT entry = entries.at(*id % max_entries);
  if (entry.id_ == *id) {
    return entry;
  } else {
    return std::nullopt;
  }
}

template <typename EntryT, typename EventType>
void retire_id_impl(
    const bool enabled,
    std::vector<EntryT>& entries,
    const size_t max_entries,
    std::mutex& mutex,
    const std::optional<size_t> id,
    const bool compute_duration) {
  if (!enabled || !id) {
    return;
  }

  bool can_compute_duration = false;
  EventType* startEvent = nullptr;
  EventType* endEvent = nullptr;
  std::optional<float> duration = std::nullopt;
  std::unique_lock<std::mutex> guard(mutex);
  EntryT* entry = &entries.at(*id % max_entries);
  if (entry->id_ == *id) {
    update_state_impl(*entry);
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
    entry = &entries.at(*id % max_entries);
    if (entry->id_ != *id) {
      LOG(INFO) << "retire_id abandoned for id " << *id
                << ", event was overwritten while waiting to compute duration.";
      return;
    }
    if (duration.has_value()) {
      entry->duration_ = duration;
    }
  }
}

template <typename EntryT>
const c10::List<c10::IValue> getCollectiveTraceImpl(
    std::vector<EntryT>& result,
    bool includeStacktraces,
    bool onlyActive) {
  auto entries = new_list();
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
}

using json = nlohmann::json;
template <typename T>
json get_dump_json(T& obj, bool includeCollectives, bool onlyActive) {
  json result;
  result[version_key_str] = version_val_str;
  result[nccl_version_key_str] = obj.nccl_version_;
  result[pg_config_key_str] = obj.getPgConfigJson();
  result[pg_status_key_str] = obj.getPgStatusJson();

  // collective trace
  if (includeCollectives) {
    std::list<json> entries;
    for (auto& e : obj.dump_entries()) {
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
}

template <typename T>
c10::Dict<c10::IValue, c10::IValue> get_dump(
    T& obj,
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.FlightRecorder__dump);
  auto result = new_dict();
  // common values
  result.insert(version_key, version_val);
  result.insert(pg_config_key, obj.getPgConfig());
  result.insert(nccl_version_key_str, obj.nccl_version_);
  result.insert(pg_status_key, obj.getPgStatus());

  // collective trace
  if (includeCollectives) {
    result.insert(
        entries_key, obj.getCollectiveTrace(includeStackTraces, onlyActive));
  }

  return result;
}

control_plane::RegisterHandler dumpHandler{
    "dump_nccl_trace_pickle",
    [](const control_plane::Request& req, control_plane::Response& res) {
      const auto& params = req.params();
      size_t validParamCount = 0;

      // valid params
      const std::string includeCollectivesStr = "includecollectives";
      const std::string includeStackTracesStr = "includestacktraces";
      const std::string onlyActiveStr = "onlyactive";

      std::unordered_map<std::string, bool> processedParams = {
          {includeCollectivesStr, true},
          {includeStackTracesStr, true},
          {onlyActiveStr, false}};

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
      res.setContent(
          dump_nccl_trace(
              processedParams[includeCollectivesStr],
              processedParams[includeStackTracesStr],
              processedParams[onlyActiveStr]),
          "application/octet-stream");
    }};

control_plane::RegisterHandler jsonDumpHandler{
    "dump_nccl_trace_json",
    [](const control_plane::Request& req, control_plane::Response& res) {
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
      res.setStatus(200);
      res.setContent(
          dump_nccl_trace_json(
              processedParams[includeCollectivesStr],
              processedParams[onlyActiveStr]),
          "application/json");
    }};

bool recursive_mkdir(const std::string& dir) {
  // Check if current dir exists
  const char* p_dir = dir.c_str();
  const bool dir_exists = (access(p_dir, F_OK) == 0);
  if (dir_exists) {
    return true;
  }

  // Find folder separator and check if we are at the top
  auto pos = dir.find_last_of("/\\");
  if (pos == std::string::npos) {
    return false;
  }

  // Try to create parent directory
  if (!(recursive_mkdir(dir.substr(0, pos)))) {
    return false;
  }

  // Try to create current directory
#ifdef _WIN32
  int ret = _mkdir(dir.c_str());
#else
  int ret = mkdir(dir.c_str(), S_IRWXU | S_IRWXG);
#endif
  // Success
  if (ret == 0) {
    return true;
  }

  // Try to create complete path again
#ifdef _WIN32
  ret = _mkdir(dir.c_str());
#else
  ret = mkdir(dir.c_str(), S_IRWXU | S_IRWXG);
#endif
  return ret == 0;
}

void DebugInfoWriter::write(const std::string& trace) {
  // Open a file for writing. The ios::binary flag is used to write data as
  // binary.
  std::ofstream file(filename_, std::ios::binary);

  // Check if the file was opened successfully.
  if (!file.is_open()) {
    LOG(ERROR) << "Error opening file for writing Flight Recorder debug info: "
               << filename_;
    return;
  }

  if (!file.write(trace.data(), static_cast<std::streamsize>(trace.size()))) {
    const auto bad = file.bad();
    LOG(ERROR) << "Error writing Flight Recorder debug info to file: "
               << filename_ << " bad bit: " << bad;
    return;
  }

  // Flush the buffer to ensure data is written to the file
  file.flush();
  if (file.bad()) {
    LOG(ERROR) << "Error flushing Flight Recorder debug info: " << filename_;
    return;
  }

  LOG(INFO) << "Finished writing Flight Recorder debug info to " << filename_;
}

DebugInfoWriter& DebugInfoWriter::getWriter(int rank) {
  if (writer_ == nullptr) {
    // Attempt to write to running user's HOME directory cache folder - if it
    // exists.
    auto homeDir = getCvarString({"HOME"}, "/tmp");
    std::string cacheDirPath = homeDir + "/.cache/torch";
    // Create the .cache directory if it doesn't exist
    recursive_mkdir(cacheDirPath);
    std::string defaultLocation = cacheDirPath + "/" + "nccl_trace_rank_";

    std::string fileNamePrefix = getCvarString(
        {"TORCH_NCCL_DEBUG_INFO_TEMP_FILE"}, defaultLocation.c_str());
    // Using std::unique_ptr here to auto-delete the writer object
    // when the pointer itself is destroyed.
    std::unique_ptr<DebugInfoWriter> writerPtr(
        new DebugInfoWriter(fileNamePrefix, rank));
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

std::optional<size_t> FlightRecorder::record(
    size_t pg_id,
    const std::tuple<std::string, std::string>& pg_name,
    size_t collective_seq_id,
    size_t p2p_seq_id,
    size_t op_id,
    std::string profiling_name,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs,
    c10::Event* start,
    c10::Event* end,
    std::chrono::milliseconds timeout_ms,
    std::shared_ptr<ProcessGroupStatus> pg_status,
    bool isP2P) {
      auto traceback =
      torch::CapturedTraceback::gather(true, true, capture_cpp_stack_);
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
    
  return recordImpl<FlightRecorder, FlightRecorder::Entry, c10::Event>(
      *this,
      std::move(te),
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

void FlightRecorder::record_accelerator_version(
    const std::string nccl_version) {
  if (!enabled_) {
    return;
  }
  std::lock_guard<std::mutex> guard(mutex_);
  nccl_version_ = std::move(nccl_version);
}

float getDurationFromEvent(c10::Event& startEvent, c10::Event& endEvent) {
  TORCH_CHECK(
      false,
      "getDuration not supported by c10::Event.")
}

void FlightRecorder::update_state(Entry& r) {
  update_state_impl(r);
}

std::vector<FlightRecorder::Entry> FlightRecorder::dump_entries() {
  return dump_entries_impl<FlightRecorder::Entry>(entries_, next_, mutex_);
}

// Returns the entry with the given id, if it exists. Otherwise, returns
// std::nullopt.
std::optional<FlightRecorder::Entry> FlightRecorder::getEntry(
    const std::optional<size_t> id) {
  return getEntryImpl<FlightRecorder::Entry>(
      entries_, enabled_, mutex_, max_entries_, id);
}

void FlightRecorder::retire_id(
    const std::optional<size_t> id,
    const bool compute_duration) {
  retire_id_impl<FlightRecorder::Entry, c10::Event>(
      enabled_, entries_, max_entries_, mutex_, id, compute_duration);
}

const c10::List<c10::IValue> FlightRecorder::getCollectiveTrace(
    bool includeStacktraces,
    bool onlyActive) {
  // Entries are returned in the order they were recorded
  auto result = dump_entries();
  return getCollectiveTraceImpl<FlightRecorder::Entry>(
      result, includeStacktraces, onlyActive);
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

std::string FlightRecorder::dump_json(
    bool includeCollectives,
    bool onlyActive) {
  return (get_dump_json<FlightRecorder>(*this, includeCollectives, onlyActive)).dump();
}

std::string FlightRecorder::dump(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  return pickle_str(get_dump<FlightRecorder>(
      *this, includeCollectives, includeStackTraces, onlyActive));
}

std::unique_ptr<DebugInfoWriter> DebugInfoWriter::writer_ = nullptr;
std::atomic<bool> DebugInfoWriter::hasWriterRegistered_(false);

#ifdef USE_C10D_NCCL

float getDurationFromEvent(Event& ncclStartEvent, Event& ncclEndEvent) {
  TORCH_CHECK(
      ncclEndEvent.query(),
      "getDuration can only be called after work is succeeded.")
  return ncclStartEvent.elapsed_time(ncclEndEvent);
}

std::optional<size_t> FlightRecorderNCCL::record(
    size_t pg_id,
    const std::tuple<std::string, std::string>& pg_name,
    size_t collective_seq_id,
    size_t p2p_seq_id,
    size_t op_id,
    std::string profiling_name,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs,
    Event* start,
    Event* end,
    std::chrono::milliseconds timeout_ms,
    std::shared_ptr<ProcessGroupStatus> pg_status,
    bool isP2P) {
      auto traceback =
      torch::CapturedTraceback::gather(true, true, capture_cpp_stack_);
      auto te = CudaEntry{
        id_,
        pg_id,
        pg_name,
        collective_seq_id,
        p2p_seq_id,
        op_id,
        std::move(profiling_name),
        std::move(traceback),
        nullptr,
        nullptr,
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
        false,
        start,
        end};

      return recordImpl<FlightRecorderNCCL, FlightRecorderNCCL::CudaEntry, Event>(
        *this,
        std::move(te),
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
}

std::string FlightRecorderNCCL::dump_json(
    const std::optional<std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::string>>>& ncclDumpMap,
    bool includeCollectives,
    bool onlyActive) {
  // Adding ncclDumpMap to the json object is NCCL specific.
  auto result =
      get_dump_json<FlightRecorderNCCL>(*this, includeCollectives, onlyActive);

  // Adding ncclDumpMap to the json object is NCCL specific.
  if (ncclDumpMap.has_value()) {
    result[nccl_comm_key_str] = ncclDumpMap.value();
  }
  return result.dump();
}

std::string FlightRecorderNCCL::dump(
    const std::optional<std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::string>>>& ncclDumpMap,
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  auto result = get_dump<FlightRecorderNCCL>(
      *this, includeCollectives, includeStackTraces, onlyActive);

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
}

void FlightRecorderNCCL::update_state(CudaEntry& r) {
  update_state_impl(r);
}

std::vector<FlightRecorderNCCL::CudaEntry> FlightRecorderNCCL::dump_entries() {
  return dump_entries_impl<FlightRecorderNCCL::CudaEntry>(entries_, next_, mutex_);
}

const c10::List<c10::IValue> FlightRecorderNCCL::getCollectiveTrace(
    bool includeStacktraces,
    bool onlyActive) {
  // Entries are returned in the order they were recorded
  auto result = dump_entries();
  return getCollectiveTraceImpl<FlightRecorderNCCL::CudaEntry>(
    result, includeStacktraces, onlyActive);
}

void FlightRecorderNCCL::retire_id(
    const std::optional<size_t> id,
    const bool compute_duration) {
  retire_id_impl<FlightRecorderNCCL::CudaEntry, Event>(
      enabled_, entries_, max_entries_, mutex_, id, compute_duration);
}
#endif // USE_C10D_NCCL
} // namespace c10d
