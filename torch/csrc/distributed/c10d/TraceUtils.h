#pragma once
#include <c10/core/ScalarType.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/irange.h>
#include <c10/util/string_view.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/profiler/combined_traceback.h>

#ifdef USE_C10D_NCCL
#include <ATen/cuda/CUDAEvent.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#endif

#include <sys/types.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace c10d {

static c10::IValue entries_key = "entries";
static c10::IValue nccl_comm_key = "nccl_comm_state";
static c10::IValue version_key = "version";
// Update whenever changing contents or formatting of the dump
// (minor when adding fields, major when changing existing fields)
static c10::IValue version_val = "2.1";
static c10::IValue pg_config_key = "pg_config";
static c10::IValue record_id_key = "record_id";
static c10::IValue pg_id_key = "pg_id";
static c10::IValue pg_name_key = "process_group";
static c10::IValue collective_seq_id_key = "collective_seq_id";
static c10::IValue p2p_seq_id_key = "p2p_seq_id";
static c10::IValue is_p2p_key = "is_p2p";
static c10::IValue op_id_key = "op_id";
static c10::IValue profiling_name_key = "profiling_name";
static c10::IValue input_sizes_key = "input_sizes";
static c10::IValue input_dtypes_key = "input_dtypes";
static c10::IValue output_sizes_key = "output_sizes";
static c10::IValue output_dtypes_key = "output_dtypes";
static c10::IValue time_created_key = "time_created_ns";
static c10::IValue duration_key = "duration_ms";

static c10::IValue frames_key = "frames";
static c10::IValue state_key = "state";
static c10::IValue line_key = "line";
static c10::IValue name_key = "name";
static c10::IValue filename_key = "filename";
static c10::IValue retired_key = "retired";
static c10::IValue time_discovered_started_key = "time_discovered_started_ns";
static c10::IValue time_discovered_completed_key =
    "time_discovered_completed_ns";

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

inline std::string get_python_cpp_trace() {
  // usage:
  // LOG(INFO) << "stacktrace: "
  //           << get_python_cpp_trace();
  // warn: might be slow in getting cpp traces
  // because of slow/broken addr2line
  // in different system libs
  std::shared_ptr<torch::CapturedTraceback> tb =
      torch::CapturedTraceback::gather(
          /*python=*/true, /*script=*/true, /*cpp=*/true);
  torch::SymbolizedTracebacks s_tbs = torch::symbolize({tb.get()});
  const auto& s_tb = s_tbs.tracebacks.at(0);
  std::stringstream oss;
  for (auto idx : c10::irange(s_tb.size())) {
    auto frame_id = s_tb[idx];
    const auto& frame = s_tbs.all_frames.at(frame_id);
    oss << "#" << idx << " " << frame.funcname << " from " << frame.filename
        << ":" << frame.lineno << std::endl;
  }
  return oss.str();
}

inline c10::Dict<c10::IValue, c10::IValue> new_dict() {
  return c10::Dict<c10::IValue, c10::IValue>(
      c10::AnyType::get(), c10::AnyType::get());
}

inline c10::List<c10::IValue> new_list() {
  return c10::List<c10::IValue>(c10::AnyType::get());
}

inline std::string ranks_str(const std::vector<uint64_t>& ranks) {
  std::string str;
  for (const auto& rank : ranks) {
    if (str.empty()) {
      str = std::to_string(rank);
    } else {
      str += ", " + std::to_string(rank);
    }
  }
  return c10::str("[", str, "]");
}

} // namespace c10d
