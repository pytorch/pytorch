#pragma once

#include <c10/util/ApproximateClock.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/profiler/combined_traceback.h>

#include <sys/types.h>

#include <cstdlib>
#include <string>
#include <system_error>
#include <vector>

namespace c10d {

/* Trace Utils Related to TORCH_NCCL_DESYNC_DEBUG */

inline std::string getTraceStartKey(const std::string& pgName, int rank) {
  return pgName + "_" + std::to_string(rank) + "_trace_start";
}

inline std::string getTraceEndKey(const std::string& pgName, int rank) {
  return pgName + "_" + std::to_string(rank) + "_trace_end";
}

inline bool traceUpdate(
    c10::intrusive_ptr<Store>& store,
    const std::string& key,
    uint64_t seq,
    const std::string& col) {
  std::vector<uint8_t> value(col.size() + sizeof(seq) + 1);
  memcpy(value.data(), &seq, sizeof(seq));
  memcpy(value.data() + sizeof(seq), col.data(), col.size());
  try {
    store->set(key, value);
    return true;
  } catch (...) {
    LOG(ERROR) << "Store is down while updating #" << seq << " with key "
               << key;
    return false;
  }
  return true;
}

enum TraceDebugEvent {
  kEventStart,
  kEventEnd,
};
// <seq, <rank, <col, start/end>>>
using TraceMap =
    std::map<uint64_t, std::map<int, std::pair<std::string, TraceDebugEvent>>>;

inline std::string ranksToString(const std::vector<int>& ranks) {
  std::string str;
  for (int rank : ranks) {
    if (str.empty()) {
      str = std::to_string(rank);
    } else {
      str += ", " + std::to_string(rank);
    }
  }
  return str;
}

inline std::string ranksFromTrace(
    const std::vector<std::pair<int, std::string>>& items) {
  std::string ranks;
  for (auto& p : items) {
    if (ranks.empty()) {
      ranks = std::to_string(p.first);
    } else {
      ranks += ", " + std::to_string(p.first);
    }
  }
  return ranks;
}

inline std::string analyzeMissingRanks(const std::vector<int>& missingRanks) {
  return c10::str(
      "\n\t - To our best knowledge, ranks [",
      ranksToString(missingRanks),
      "] are the lagging ranks that caused this timeout. "
      "They never joined any collectives");
}

inline std::string analyzeLaggingRanks(const TraceMap& traceMap) {
  uint64_t lagSeq = traceMap.begin()->first;
  std::vector<int> startRanks;
  std::vector<int> endRanks;
  for (auto& p : traceMap.begin()->second) {
    if (p.second.second == kEventStart) {
      startRanks.push_back(p.first);
    } else {
      endRanks.push_back(p.first);
    }
  }
  std::string report =
      "\n\t - To our best knowledge, the lagging/dead/mismatched ranks "
      "that caused the desync are:";
  if (startRanks.size()) {
    report += c10::str(
        "\n\t   - [",
        ranksToString(startRanks),
        "] joined but didn't finish collective #",
        lagSeq,
        " (count from 1)");
  }
  if (endRanks.size()) {
    report += c10::str(
        "\n\t     [",
        ranksToString(endRanks),
        "] finished collective #",
        lagSeq,
        ", but didn't join collective #",
        lagSeq + 1,
        " (count from 1)");
  }
  return report;
}

inline std::string dumpSnapshot(TraceMap& traceMap) {
  std::string report = "\n\t - Snapshot of ranks' latest states:";
  for (auto& tracePair : traceMap) {
    uint64_t seq = tracePair.first;
    std::map<int, std::pair<std::string, TraceDebugEvent>>& subMap =
        tracePair.second;

    std::unordered_map<std::string, std::vector<int>> collectivesStart;
    std::unordered_map<std::string, std::vector<int>> collectivesEnd;
    for (auto& p : subMap) {
      int rank = p.first;
      const std::string& col = p.second.first;
      if (p.second.second == kEventStart) {
        collectivesStart[col].push_back(rank);
      } else {
        collectivesEnd[col].push_back(rank);
      }
    }

    if (collectivesStart.size()) {
      report += c10::str("\n\t   #", seq, " started ranks:");
      for (auto& mapPair : collectivesStart) {
        report += c10::str(
            "\n\t     [",
            ranksToString(mapPair.second),
            "] started ",
            mapPair.first);
      }
    }
    if (collectivesEnd.size()) {
      report += c10::str("\n\t   #", seq, " finished ranks:");
      for (auto& mapPair : collectivesEnd) {
        report += c10::str(
            "\n\t     [",
            ranksToString(mapPair.second),
            "] finished ",
            mapPair.first);
      }
    }
  }
  return report;
}

inline bool parseTraceValue(
    c10::intrusive_ptr<Store>& store,
    const std::string& key,
    uint64_t& seq,
    std::string& col) {
  try {
    std::vector<uint8_t> traceValue = store->get(key);
    memcpy(&seq, traceValue.data(), sizeof(seq));
    std::string colName((char*)traceValue.data() + sizeof(seq));
    col = colName;
    return true;
  } catch (...) {
    LOG(ERROR) << "Store is down while getting key " << key;
    return false;
  }
  return true;
}

inline std::string retrieveDesyncReport(
    c10::intrusive_ptr<Store>& store,
    const std::string& pgName,
    int myRank,
    int worldSize) {
  std::string report;

  uint64_t thisSeq;
  std::string thisCol;

  std::vector<int> missingRanks;
  TraceMap traceMap;

  for (const auto rank : c10::irange(worldSize)) {
    // Build traceMapStart.
    uint64_t seqStart;
    {
      std::string traceKeyStart = getTraceStartKey(pgName, rank);
      if (!store->check({traceKeyStart})) {
        missingRanks.push_back(rank);
        continue;
      }
      std::string col;
      if (!parseTraceValue(store, traceKeyStart, seqStart, col)) {
        return report;
      }
      traceMap[seqStart].emplace(rank, std::make_pair(col, kEventStart));
      if (rank == myRank) {
        thisSeq = seqStart;
        thisCol = std::move(col);
      }
    }

    // Build traceMapEnd.
    {
      std::string traceKeyEnd = getTraceEndKey(pgName, rank);
      if (!store->check({traceKeyEnd})) {
        continue;
      }
      uint64_t seq;
      std::string col;
      if (!parseTraceValue(store, traceKeyEnd, seq, col)) {
        return report;
      }
      if (seq == seqStart) {
        traceMap[seq][rank].second = kEventEnd;
      }
    }
  }

  TORCH_INTERNAL_ASSERT(
      !missingRanks.empty() || !traceMap.empty(),
      "Trace shouldn't be empty while enabled GLOO_ASYNC_TIMEOUT_DEBUG");
  TORCH_INTERNAL_ASSERT(
      !thisCol.empty(),
      "Timeout rank [",
      myRank,
      "] must have collective tracking iteam in c10::Store trace");
  TORCH_INTERNAL_ASSERT(
      traceMap[thisSeq][myRank].second == kEventStart,
      "Timeout rank [",
      myRank,
      "] last trace item must be kEventStart. thisSeq = ",
      thisSeq,
      ", col = ",
      thisCol);

  report += c10::str(
      "\n\t - [", myRank, "] Timeout at collective: ", thisCol, ", #", thisSeq);

  if (!missingRanks.empty()) {
    report += analyzeMissingRanks(missingRanks);
  } else {
    report += analyzeLaggingRanks(traceMap);
    report += dumpSnapshot(traceMap);
  }

  return report;
}

/* Trace Utils Related to Flight Recorder */

/* Note: this is only used by PGNCCL (could be generalized in an ideal world but
 * wasn't done that way, so isn't expected to be fully general at the moment) */

#ifdef USE_C10D_NCCL

DebugInfoWriter::DebugInfoWriter(int rank) {
  std::string fileName = getCvarString(
      {"TORCH_NCCL_DEBUG_INFO_TEMP_FILE"}, "/tmp/nccl_trace_rank_");
  filename_ = c10::str(fileName, rank);
}

DebugInfoWriter::~DebugInfoWriter() = default;

void DebugInfoWriter::write(const std::string& ncclTrace) {
  // Open a file for writing. The ios::binary flag is used to write data as
  // binary.
  std::ofstream file(filename_, std::ios::binary);

  // Check if the file was opened successfully.
  if (!file.is_open()) {
    LOG(ERROR) << "Error opening file for writing NCCLPG debug info: "
               << filename_;
    return;
  }

  file.write(ncclTrace.data(), ncclTrace.size());
  LOG(INFO) << "Finished writing NCCLPG debug info to " << filename_;
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

inline c10::Dict<c10::IValue, c10::IValue> new_dict() {
  return c10::Dict<c10::IValue, c10::IValue>(
      c10::AnyType::get(), c10::AnyType::get());
}

inline c10::List<c10::IValue> new_list() {
  return c10::List<c10::IValue>(c10::AnyType::get());
}

struct NCCLTraceBuffer {
  static NCCLTraceBuffer* get() {
    // intentionally leak on exit
    // because this will hold python state that may get destructed
    static NCCLTraceBuffer* instance = new NCCLTraceBuffer();
    return instance;
  }
  NCCLTraceBuffer() {
    max_entries_ = getCvarInt({"TORCH_NCCL_TRACE_BUFFER_SIZE"}, 0);
    capture_cpp_stack_ = getCvarBool({"TORCH_NCCL_TRACE_CPP_STACK"}, false);
    enabled_ = max_entries_ > 0;
  }
  using EventList = std::vector<at::cuda::CUDAEvent>;
  struct Entry {
    size_t id_; // incremented id in the trace buffer
                // used to figure out where in the circular entries
                // buffer this entry will be located to
                // update state information
    size_t pg_id_;
    size_t seq_id_; // as tracked by the process group
    const char* profiling_name_;

    std::shared_ptr<torch::CapturedTraceback> traceback_;
    // we borrow pointser to start_ and end_ so we can query the state
    // on reporting. However, once the event is completed, the call
    // to `complete` will clear these.
    EventList *start_, *end_;

    // timestamp when the entry was created, likely close to the time the work
    // was 'enqueued'- not necessarily started
    c10::time_t time_created_;

    const char* state_ = "scheduled";

    // size information for input/output tensors
    c10::SmallVector<int, 4> input_dims_;
    c10::SmallVector<int, 4> output_dims_;
    c10::SmallVector<int64_t, 8> sizes_; // flattened from inputs, outputs
    bool retired_ = false; // is this work entry no longer in the workMetaList_?
                           // a retired but not completed event has timed out
  };

  bool enabled_ = false;
  bool capture_cpp_stack_ = false;
  std::mutex mutex_;
  std::vector<Entry> entries_;
  size_t max_entries_ = 0;
  size_t next_ = 0;
  size_t id_ = 0;

  c10::optional<size_t> record(
      size_t pg_id,
      size_t seq_id,
      const char* profiling_name,
      const std::vector<at::Tensor>& inputs,
      const std::vector<at::Tensor>& outputs,
      EventList* start,
      EventList* end) {
    if (!enabled_) {
      return c10::nullopt;
    }
    auto traceback =
        torch::CapturedTraceback::gather(true, true, capture_cpp_stack_);
    std::lock_guard<std::mutex> guard(mutex_);

    auto te = Entry{
        id_,
        pg_id,
        seq_id,
        profiling_name,
        std::move(traceback),
        std::move(start),
        std::move(end),
        c10::getTime()};

    for (const auto& input : inputs) {
      c10::IntArrayRef sizes = input.sizes();
      te.input_dims_.push_back(sizes.size());
      te.sizes_.insert(te.sizes_.end(), sizes.begin(), sizes.end());
    }

    for (const auto& output : outputs) {
      c10::IntArrayRef sizes = output.sizes();
      te.output_dims_.push_back(sizes.size());
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
  }

  void update_state(Entry& r) {
    if (r.start_ != nullptr) {
      bool started = true;
      for (auto& ev : *r.start_) {
        if (!ev.query()) {
          started = false;
          break;
        }
      }
      if (started) {
        r.state_ = "started";
      }
    }
    if (r.end_ != nullptr) {
      bool completed = true;
      for (auto& ev : *r.end_) {
        if (!ev.query()) {
          completed = false;
          break;
        }
      }
      if (completed) {
        r.state_ = "completed";
      }
    }
  }

  std::vector<Entry> dump_entries() {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<Entry> result;
    result.reserve(entries_.size());
    result.insert(result.end(), entries_.begin() + next_, entries_.end());
    result.insert(result.end(), entries_.begin(), entries_.begin() + next_);
    // query any remaining events
    for (auto& r : result) {
      update_state(r);
      r.start_ = r.end_ = nullptr;
    }
    return result;
  }

  void retire_id(c10::optional<size_t> id) {
    if (!enabled_ || !id) {
      return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    auto& entry = entries_.at(*id % max_entries_);
    if (entry.id_ == *id) {
      update_state(entry);
      entry.retired_ = true;
      entry.start_ = entry.end_ = nullptr;
    }
  }

  std::string dump() {
    auto result = dump_entries();
    auto entries = new_list();
    c10::IValue pg_id_s = "pg_id";
    c10::IValue seq_id_s = "seq_id";
    c10::IValue profiling_name_s = "profiling_name";
    c10::IValue input_sizes_s = "input_sizes";
    c10::IValue output_sizes_s = "output_sizes";
    c10::IValue time_created_s = "time_created_us";

    c10::IValue frames_s = "frames";
    c10::IValue state_s = "state";
    c10::IValue line_s = "line";
    c10::IValue name_s = "name";
    c10::IValue filename_s = "filename";
    c10::IValue retired_s = "retired";

    std::vector<torch::CapturedTraceback*> tracebacks;
    for (auto& e : result) {
      tracebacks.push_back(e.traceback_.get());
    }
    torch::SymbolizedTracebacks stracebacks = torch::symbolize(tracebacks);
    std::vector<c10::IValue> all_frames;
    for (const auto& f : stracebacks.all_frames) {
      auto d = new_dict();
      d.insert(name_s, f.funcname);
      d.insert(filename_s, f.filename);
      d.insert(line_s, int64_t(f.lineno));
      all_frames.emplace_back(std::move(d));
    }

    for (auto i : c10::irange(result.size())) {
      auto& e = result.at(i);
      auto& tb = stracebacks.tracebacks.at(i);
      auto dict = new_dict();
      dict.insert(pg_id_s, int64_t(e.pg_id_));
      dict.insert(seq_id_s, int64_t(e.seq_id_));
      dict.insert(profiling_name_s, e.profiling_name_);
      dict.insert(time_created_s, int64_t(e.time_created_ / 1000));

      auto it = e.sizes_.begin();
      auto read_sizes = [&](const c10::SmallVector<int, 4>& dims) {
        auto sizes = new_list();
        for (auto dim : dims) {
          auto arg_sizes = new_list();
          for (auto i : c10::irange(dim)) {
            (void)i;
            arg_sizes.push_back(*it++);
          }
          sizes.push_back(arg_sizes);
        }
        return sizes;
      };

      dict.insert(input_sizes_s, read_sizes(e.input_dims_));
      dict.insert(output_sizes_s, read_sizes(e.output_dims_));
      dict.insert(state_s, e.state_);
      dict.insert(retired_s, e.retired_);

      auto frames = new_list();
      for (int64_t frame : tb) {
        frames.push_back(all_frames.at(frame));
      }
      dict.insert(frames_s, frames);
      entries.push_back(dict);
    }
    return pickle_str(entries);
  }
};

#endif
} // namespace c10d
