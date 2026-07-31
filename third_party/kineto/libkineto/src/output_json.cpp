/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "output_json.h"

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iterator>
#include <string_view>
#include "Config.h"
#include "EnvMetadata.h"
#include "TraceSpan.h"

#include "Logger.h"

namespace KINETO_NAMESPACE {
namespace {

constexpr int kSchemaVersion = 1;
constexpr char kFlowStart = 's';
constexpr char kFlowEnd = 'f';

// CPU op name that is used to store collectives metadata
// TODO: share the same string across c10d, profiler and libkineto
constexpr std::string_view kParamCommsCallName = "record_param_comms";
// Collective function metadata populated from CPU op to GPU kernel
constexpr std::string_view kCollectiveName = "Collective name";
constexpr std::string_view kDtype = "dtype";
constexpr std::string_view kInMsgNelems = "In msg nelems";
constexpr std::string_view kOutMsgNelems = "Out msg nelems";
constexpr std::string_view kGroupSize = "Group size";
constexpr std::string_view kInSplit = "In split size";
constexpr std::string_view kOutSplit = "Out split size";
constexpr std::string_view kProcessGroupName = "Process Group Name";
constexpr std::string_view kProcessGroupDesc = "Process Group Description";
constexpr std::string_view kGroupRanks = "Process Group Ranks";
constexpr std::string_view kInTensorsStart = "Input Tensors start";
constexpr std::string_view kOutTensorsStart = "Output Tensors start";
constexpr std::string_view kRank = "Rank";
constexpr std::string_view kP2pSrc = "Src Rank";
constexpr std::string_view kP2pDst = "Dst Rank";
constexpr std::string_view kSeqNum = "Seq";
constexpr std::string_view kCommsId = "Comms Id";

// Collective string metadata arrives quoted from the legacy RawJson path and
// unquoted from the typed path; strip a single surrounding pair of double
// quotes so downstream emission can re-quote uniformly (tolerates both).
std::string stripQuotes(std::string s) {
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
    s.pop_back();
    s.erase(0, 1);
  }
  return s;
}

#ifdef __linux__
constexpr std::string_view kDefaultLogFileFmt =
    "/tmp/libkineto_activities_{}.json";
#else
constexpr std::string_view kDefaultLogFileFmt = "libkineto_activities_{}.json";
#endif

void sanitizeStrForJSON(std::string& value) {
  // Replace all backslashes with forward slash because Windows paths causing
  // JSONDecodeError.
  std::ranges::replace(value, '\\', '/');
  // Remove all new line characters
  std::erase(value, '\n');
}

// Escape bare double quotes so an event name can be safely embedded in a JSON
// string value ("name": "<value>"). Names can contain `"` (e.g.
// dynamo-generated co_filenames like {"device": "cpu"} surfaced with
// with_stack=True); an unescaped `"` corrupts the whole trace. Run after
// sanitizeStrForJSON() so the inserted backslashes aren't rewritten. See
// pytorch/pytorch#146900.
void escapeQuotesForJSON(std::string& value) {
  for (size_t pos = value.find('"'); pos != std::string::npos;
       pos = value.find('"', pos + 2)) {
    value.insert(pos, 1, '\\');
  }
}

std::string string2hex(const std::string& str) {
  std::string out;
  out.reserve(str.size() * 2);
  for (uint8_t c : str) {
    // “:02x” -> two‐digit, zero‐padded, lowercase hex
    fmt::format_to(std::back_inserter(out), "{:02x}", c);
  }
  return out;
}

void sanitizeForNonReadableChars(std::string& value) {
  for (auto& c : value) {
    if (!std::isprint(c)) {
      LOG(WARNING) << "Non JSON compliant character found in string: 0x"
                   << string2hex(value) << " Replacing with 'unknown'";
      value = "unknown";
      break;
    }
  }
}

inline int64_t sanitizeTid(int64_t tid) {
  // Convert all negative tids to its positive value. Create a specific case
  // for INT64_MIN so it is obvious how it is being handled.
  if (tid == INT64_MIN) {
    return 0;
  }
  return std::abs(tid);
}

// Format nanosecond timestamp as "us.fractional" for Chrome Trace JSON.
std::string fmtTs(int64_t time_ns) {
  return fmt::format("{}.{:03}", time_ns / 1000, time_ns % 1000);
}

bool isWhitespace(std::string_view s) {
  return std::ranges::all_of(
      s, [](unsigned char c) { return std::isspace(c); });
}

} // namespace

ChromeTraceBaseTime& ChromeTraceBaseTime::singleton() {
  static ChromeTraceBaseTime instance;
  return instance;
}

// The 'ts' field written into the json file has 19 significant digits,
// while a double can only represent 15-16 digits. By using relative time,
// other applications can accurately read the 'ts' field as a double.
// Use the program loading time as the baseline time.
int64_t transToRelativeTime(int64_t time) {
  // Sometimes after converting to relative time, it can be a few nanoseconds
  // negative. Since Chrome trace and json processing will throw a parser error,
  // guard this.
  int64_t res = time - ChromeTraceBaseTime::singleton().get();
  if (res < 0) {
    return 0;
  }
  return res;
}

// Internal helper for building the "args" JSON object content.
// Handles comma separation automatically.
class ArgsBuilder {
 public:
  // Add a key with a pre-formatted value (number, boolean, already-quoted
  // string, JSON array/object).
  void addRaw(std::string_view key, std::string_view value) {
    appendComma();
    fmt::format_to(std::back_inserter(buf_), R"("{}": {})", key, value);
  }

  // Add a key with a string value (will be JSON-quoted).
  void addQuoted(std::string_view key, std::string_view value) {
    appendComma();
    fmt::format_to(std::back_inserter(buf_), R"("{}": "{}")", key, value);
  }

  // Append a pre-formatted JSON fragment (e.g., from metadataJson()).
  void appendFragment(std::string_view json_fragment) {
    // Skip empty or whitespace-only fragments: appending them after a comma
    // from appendComma() would produce invalid JSON (e.g., "key": value, }).
    if (json_fragment.empty() || isWhitespace(json_fragment)) {
      return;
    }
    appendComma();
    buf_.append(json_fragment);
  }

  [[nodiscard]] std::string_view str() const {
    return buf_;
  }

  [[nodiscard]] bool empty() const {
    return buf_.empty();
  }

  // Return the full "args" JSON field (e.g., ', "args": { ... }'), or an
  // empty string if no args were added.
  [[nodiscard]] std::string renderField() const {
    if (buf_.empty()) {
      return {};
    }
    return fmt::format(
        R"JSON(,
    "args": {{
      {}
    }})JSON",
        buf_);
  }

 private:
  void appendComma() {
    if (!buf_.empty()) {
      buf_.append(",");
    }
  }
  std::string buf_;
};

void ChromeTraceLogger::writeMetadataEvent(
    std::string_view name,
    int64_t ts,
    std::string_view pid,
    std::string_view tid,
    std::string_view arg_key,
    std::string_view arg_value) {
  // clang-format off
  fmt::print(traceOf_, R"JSON(
  {{
    "ph": "M",
    "name": "{}",
    "ts": {},
    "pid": {},
    "tid": {},
    "args": {{
      "{}": {}
    }}
  }},)JSON",
      name, fmtTs(ts), pid, tid, arg_key, arg_value);
  // clang-format on
}

void ChromeTraceLogger::writeCompleteEvent(
    std::string_view cat,
    std::string_view name,
    std::string_view pid,
    std::string_view tid,
    int64_t ts,
    int64_t dur,
    const ArgsBuilder& args) {
  // clang-format off
  fmt::print(traceOf_, R"JSON(
  {{
    "ph": "X",
    "cat": "{}",
    "name": "{}",
    "pid": {},
    "tid": {},
    "ts": {},
    "dur": {}
    {}
  }},)JSON",
      cat, name, pid, tid, fmtTs(ts), fmtTs(dur), args.renderField());
  // clang-format on
}

void ChromeTraceLogger::writeInstantEvent(
    std::string_view cat,
    std::string_view name,
    std::string_view scope,
    std::string_view pid,
    std::string_view tid,
    int64_t ts,
    const ArgsBuilder& args,
    bool finalEvent) {
  std::string cat_str;
  if (!cat.empty()) {
    cat_str = fmt::format(
        R"JSON(
    "cat": "{}",)JSON",
        cat);
  }
  const char* trailing = finalEvent ? "" : ",";
  // clang-format off
  fmt::print(traceOf_, R"JSON(
  {{
    "ph": "i",
    {}
    "s": "{}",
    "name": "{}",
    "pid": {},
    "tid": {},
    "ts": {}
    {}
  }}{})JSON",
      cat_str, scope, name, pid, tid, fmtTs(ts), args.renderField(), trailing);
  // clang-format on
}

void ChromeTraceLogger::writeCounterEvent(
    std::string_view cat,
    std::string_view name,
    std::string_view pid,
    std::string_view tid,
    int64_t ts,
    const ArgsBuilder& args) {
  // clang-format off
  fmt::print(traceOf_, R"JSON(
  {{
    "ph": "C",
    "cat": "{}",
    "name": "{}",
    "pid": {},
    "tid": {},
    "ts": {},
    "args": {{
      {}
    }}
  }},)JSON",
      cat, name, pid, tid, fmtTs(ts), args.str());
  // clang-format on
}

void ChromeTraceLogger::writeFlowEvent(
    char type,
    int64_t id,
    std::string_view pid,
    std::string_view tid,
    int64_t ts,
    std::string_view cat,
    std::string_view name) {
  // Flow events must bind to specific slices in order to exist.
  // Only Flow end needs to specify a binding point to enclosing slice.
  // Flow start automatically sets binding point to enclosing slice.
  const auto* const binding = (type == kFlowEnd) ? R"(, "bp": "e")" : "";

  // clang-format off
  fmt::print(traceOf_, R"JSON(
  {{
    "ph": "{}",
    "id": {},
    "pid": {},
    "tid": {},
    "ts": {},
    "cat": "{}",
    "name": "{}"
    {}
  }},)JSON",
      type, id, pid, tid, fmtTs(ts), cat, name, binding);
  // clang-format on
}

// Integer overloads — delegate to the string_view versions.
void ChromeTraceLogger::writeMetadataEvent(
    std::string_view name,
    int64_t ts,
    int64_t pid,
    int64_t tid,
    std::string_view arg_key,
    std::string_view arg_value) {
  writeMetadataEvent(
      name, ts, std::to_string(pid), std::to_string(tid), arg_key, arg_value);
}

void ChromeTraceLogger::writeCompleteEvent(
    std::string_view cat,
    std::string_view name,
    int64_t pid,
    int64_t tid,
    int64_t ts,
    int64_t dur,
    const ArgsBuilder& args) {
  writeCompleteEvent(
      cat, name, std::to_string(pid), std::to_string(tid), ts, dur, args);
}

void ChromeTraceLogger::writeInstantEvent(
    std::string_view cat,
    std::string_view name,
    std::string_view scope,
    int64_t pid,
    int64_t tid,
    int64_t ts,
    const ArgsBuilder& args,
    bool finalEvent) {
  writeInstantEvent(
      cat,
      name,
      scope,
      std::to_string(pid),
      std::to_string(tid),
      ts,
      args,
      finalEvent);
}

void ChromeTraceLogger::writeCounterEvent(
    std::string_view cat,
    std::string_view name,
    int64_t pid,
    int64_t tid,
    int64_t ts,
    const ArgsBuilder& args) {
  writeCounterEvent(
      cat, name, std::to_string(pid), std::to_string(tid), ts, args);
}

void ChromeTraceLogger::writeFlowEvent(
    char type,
    int64_t id,
    int64_t pid,
    int64_t tid,
    int64_t ts,
    std::string_view cat,
    std::string_view name) {
  writeFlowEvent(
      type, id, std::to_string(pid), std::to_string(tid), ts, cat, name);
}

void ChromeTraceLogger::metadataToJSON(
    const std::unordered_map<std::string, std::string>& metadata) {
  for (const auto& [k, v] : metadata) {
    std::string sanitizedValue = v;
    // There is a separate mechanism for recording distributedInfo in on-demand
    // so add a guard to prevent "double counting" in auto-trace.
    if (k == "distributedInfo") {
      distInfo_.distInfo_present_ = true;
    }
    sanitizeStrForJSON(sanitizedValue);
    fmt::print(
        traceOf_,
        R"JSON(
      "{}": {},)JSON",
        k,
        sanitizedValue);
  }
}

std::unordered_map<std::string, std::string> ChromeTraceLogger::
    addEnvVarsToMetadata(
        const std::unordered_map<std::string, std::string>& metadata) {
  // Get environment metadata from the EnvMetadata module
  auto combined = libkineto::getEnvMetadata();
  // Original metadata takes precedence
  for (const auto& [k, v] : metadata) {
    combined[k] = v;
  }
  return combined;
}

void ChromeTraceLogger::handleTraceStart(
    const std::unordered_map<std::string, std::string>& metadata,
    const std::string& device_properties) {
  if (!traceOf_) {
    return;
  }
  std::string display_unit = "ms";
#ifdef DISPLAY_TRACE_IN_NS
  display_unit = "ns";
#endif
  // clang-format off
  fmt::print(traceOf_, R"JSON( {{
    "schemaVersion": {},
    "deviceProperties": [{}],
    )JSON",
      kSchemaVersion, device_properties);
  // clang-format on

  // Note that metadataToJSON writes to the trace file.
  auto combinedMetadata = addEnvVarsToMetadata(metadata);
  metadataToJSON(combinedMetadata);

  // Note that we open the `traceEvents` array, but we do not close it. Most
  // of the rest of the writing are writing new trace events to the array. We
  // close the array in `finalizeTrace`.
  // clang-format off
  fmt::print(traceOf_, R"JSON(
    "displayTimeUnit": "{}",
    "baseTimeNanoseconds": {},
    "traceEvents": [
    )JSON",
      display_unit,
      ChromeTraceBaseTime::singleton().get());
  // clang-format on
}

static std::string defaultFileName() {
  return fmt::format(kDefaultLogFileFmt, processId());
}

void ChromeTraceLogger::openTraceFile() {
  tempFileName_ = fileName_ + ".tmp";
  traceOf_.open(tempFileName_, std::ofstream::out | std::ofstream::trunc);
  if (!traceOf_) {
    PLOG(ERROR) << "Failed to open '" << fileName_ << "'";
  } else {
    LOG(INFO) << "Tracing to temporary file " << fileName_;
  }
}

void ChromeTraceLogger::finalizeMemoryTrace(
    [[maybe_unused]] const std::string& url,
    [[maybe_unused]] const Config& config) {
  LOG(INFO) << "finalizeMemoryTrace not implemented for ChromeTraceLogger";
}

ChromeTraceLogger::ChromeTraceLogger(const std::string& traceFileName) {
  fileName_ = traceFileName.empty() ? defaultFileName() : traceFileName;
  traceOf_.clear(std::ios_base::badbit);
  openTraceFile();
}

void ChromeTraceLogger::handleDeviceInfo(const DeviceInfo& info, int64_t time) {
  if (!traceOf_) {
    return;
  }

  time = transToRelativeTime(time);
  writeMetadataEvent(
      /*name=*/"process_name",
      /*ts=*/time,
      /*pid=*/info.id,
      /*tid=*/0,
      /*arg_key=*/"name",
      /*arg_value=*/fmt::format("\"{}\"", info.name));
  writeMetadataEvent(
      /*name=*/"process_labels",
      /*ts=*/time,
      /*pid=*/info.id,
      /*tid=*/0,
      /*arg_key=*/"labels",
      /*arg_value=*/fmt::format("\"{}\"", info.label));
  writeMetadataEvent(
      /*name=*/"process_sort_index",
      /*ts=*/time,
      /*pid=*/info.id,
      /*tid=*/0,
      /*arg_key=*/"sort_index",
      /*arg_value=*/fmt::format("{}", info.sortIndex));
}

void ChromeTraceLogger::handleResourceInfo(
    const ResourceInfo& info,
    int64_t time) {
  if (!traceOf_) {
    return;
  }

  time = transToRelativeTime(time);
  int64_t tid = sanitizeTid(info.id);
  writeMetadataEvent(
      /*name=*/"thread_name",
      /*ts=*/time,
      /*pid=*/info.deviceId,
      /*tid=*/tid,
      /*arg_key=*/"name",
      /*arg_value=*/fmt::format("\"{}\"", info.name));
  writeMetadataEvent(
      /*name=*/"thread_sort_index",
      /*ts=*/time,
      /*pid=*/info.deviceId,
      /*tid=*/tid,
      /*arg_key=*/"sort_index",
      /*arg_value=*/fmt::format("{}", info.sortIndex));
}

void ChromeTraceLogger::handleOverheadInfo(
    const OverheadInfo& info,
    int64_t time) {
  if (!traceOf_) {
    return;
  }

  // TOOD: reserve pid = -1 for overhead but we need to rethink how to scale
  // this for other metadata
  time = transToRelativeTime(time);
  writeMetadataEvent(
      /*name=*/"process_name",
      /*ts=*/time,
      /*pid=*/-1,
      /*tid=*/0,
      /*arg_key=*/"name",
      /*arg_value=*/fmt::format("\"{}\"", info.name));
  writeMetadataEvent(
      /*name=*/"process_sort_index",
      /*ts=*/time,
      /*pid=*/-1,
      /*tid=*/0,
      /*arg_key=*/"sort_index",
      /*arg_value=*/fmt::format("{}", 0x100000All));
}

void ChromeTraceLogger::handleTraceSpan(const TraceSpan& span) {
  if (!traceOf_) {
    return;
  }

  int64_t start = transToRelativeTime(span.startTime);

  // If endTime is 0 and start time is non-zero, dur can overflow. Add
  // a guard to prevent this.
  int64_t dur = (span.endTime == 0) ? 0 : span.endTime - span.startTime;

  ArgsBuilder args;
  args.addRaw("Op count", fmt::format("{}", span.opCount));
  writeCompleteEvent(
      /*cat=*/"Trace",
      /*name=*/fmt::format("{}{} ({})", span.prefix, span.name, span.iteration),
      /*pid=*/R"("Spans")",
      /*tid=*/fmt::format("\"{}\"", span.name),
      /*ts=*/start,
      /*dur=*/dur,
      /*args=*/args);
  writeMetadataEvent(
      /*name=*/"process_sort_index",
      /*ts=*/start,
      /*pid=*/R"("Spans")",
      /*tid=*/"0",
      /*arg_key=*/"sort_index",
      // Large sort index to appear at the bottom
      /*arg_value=*/fmt::format("{}", 0x20000000ll));

  addIterationMarker(span);
}

void ChromeTraceLogger::addIterationMarker(const TraceSpan& span) {
  if (!traceOf_) {
    return;
  }

  int64_t start = transToRelativeTime(span.startTime);

  ArgsBuilder args;
  writeInstantEvent(
      /*cat=*/"",
      /*name=*/fmt::format("Iteration Start: {}", span.name),
      /*scope=*/"g",
      /*pid=*/R"("Traces")",
      /*tid=*/fmt::format("\"Trace {}\"", span.name),
      /*ts=*/start,
      /*args=*/args);
}

void ChromeTraceLogger::handleGenericInstantEvent(
    const libkineto::ITraceActivity& op) {
  if (!traceOf_) {
    return;
  }

  int64_t ts = transToRelativeTime(op.timestamp());
  ArgsBuilder args;
  args.appendFragment(op.metadataJson());
  writeInstantEvent(
      /*cat=*/toString(op.type()),
      /*name=*/op.name(),
      /*scope=*/"t",
      /*pid=*/op.deviceId(),
      /*tid=*/sanitizeTid(op.resourceId()),
      /*ts=*/ts,
      /*args=*/args);
}

void ChromeTraceLogger::handleCounterEvent(
    const libkineto::ITraceActivity& op) {
  if (!traceOf_) {
    return;
  }

  ArgsBuilder args;
  for (const auto& [name, value] : op.counterValues()) {
    args.addRaw(name, fmt::format("{}", value));
  }

  int64_t ts = transToRelativeTime(op.timestamp());
  writeCounterEvent(
      /*cat=*/toString(op.type()),
      /*name=*/op.name(),
      /*pid=*/op.deviceId(),
      /*tid=*/sanitizeTid(op.resourceId()),
      /*ts=*/ts,
      /*args=*/args);
}

void ChromeTraceLogger::appendCollectiveArgs(
    ArgsBuilder& args,
    const ITraceActivity& collectiveRecord) {
  auto collectiveName = stripQuotes(
      collectiveRecord.getMetadataValue(std::string(kCollectiveName)));
  const auto& inMsgSize =
      collectiveRecord.getMetadataValue(std::string(kInMsgNelems));
  const auto& outMsgSize =
      collectiveRecord.getMetadataValue(std::string(kOutMsgNelems));
  const auto& groupSize =
      collectiveRecord.getMetadataValue(std::string(kGroupSize));
  auto dtype =
      stripQuotes(collectiveRecord.getMetadataValue(std::string(kDtype)));
  if (!collectiveName.empty() && !inMsgSize.empty() && !outMsgSize.empty() &&
      !groupSize.empty() && !dtype.empty()) {
    args.addQuoted(kCollectiveName, collectiveName);
    args.addRaw(kInMsgNelems, inMsgSize);
    args.addRaw(kOutMsgNelems, outMsgSize);
    args.addRaw(kGroupSize, groupSize);
    args.addQuoted(kDtype, dtype);
  }

  auto inputTensorStarts = stripQuotes(
      collectiveRecord.getMetadataValue(std::string(kInTensorsStart)));
  auto outputTensorStarts = stripQuotes(
      collectiveRecord.getMetadataValue(std::string(kOutTensorsStart)));
  if (!inputTensorStarts.empty()) {
    args.addQuoted(kInTensorsStart, inputTensorStarts);
  }
  if (!outputTensorStarts.empty()) {
    args.addQuoted(kOutTensorsStart, outputTensorStarts);
  }

  // In/out split size are valid for all_to_all
  auto inSplitSize =
      stripQuotes(collectiveRecord.getMetadataValue(std::string(kInSplit)));
  auto outSplitSize =
      stripQuotes(collectiveRecord.getMetadataValue(std::string(kOutSplit)));
  if (!inSplitSize.empty() && !outSplitSize.empty()) {
    args.addQuoted(kInSplit, inSplitSize);
    args.addQuoted(kOutSplit, outSplitSize);
  }

  auto processGroupName = stripQuotes(
      collectiveRecord.getMetadataValue(std::string(kProcessGroupName)));
  if (!processGroupName.empty()) {
    args.addQuoted(kProcessGroupName, processGroupName);
  }

  auto processGroupDesc = stripQuotes(
      collectiveRecord.getMetadataValue(std::string(kProcessGroupDesc)));
  if (!processGroupDesc.empty()) {
    args.addQuoted(kProcessGroupDesc, processGroupDesc);
  }

  auto groupRanks =
      stripQuotes(collectiveRecord.getMetadataValue(std::string(kGroupRanks)));
  if (!groupRanks.empty()) {
    args.addQuoted(kGroupRanks, groupRanks);
  }

  const auto& dstRank = collectiveRecord.getMetadataValue(std::string(kP2pDst));
  const auto& srcRank = collectiveRecord.getMetadataValue(std::string(kP2pSrc));
  if (!dstRank.empty()) {
    args.addRaw(kP2pDst, dstRank);
  }
  if (!srcRank.empty()) {
    args.addRaw(kP2pSrc, srcRank);
  }

  const auto& seqNum = collectiveRecord.getMetadataValue(std::string(kSeqNum));
  if (!seqNum.empty()) {
    args.addRaw(kSeqNum, seqNum);
  }

  const auto& commsId =
      collectiveRecord.getMetadataValue(std::string(kCommsId));
  if (!commsId.empty()) {
    args.addRaw(kCommsId, commsId);
  }
}

void ChromeTraceLogger::appendCollectiveMetadata(
    ArgsBuilder& args,
    const ITraceActivity& collectiveRecord,
    const std::string& backend,
    const std::string& backendConfig) {
  appendCollectiveArgs(args, collectiveRecord);

  const auto& groupSize =
      collectiveRecord.getMetadataValue(std::string(kGroupSize));
  auto processGroupName = stripQuotes(
      collectiveRecord.getMetadataValue(std::string(kProcessGroupName)));
  auto processGroupDesc = stripQuotes(
      collectiveRecord.getMetadataValue(std::string(kProcessGroupDesc)));
  auto groupRanks =
      stripQuotes(collectiveRecord.getMetadataValue(std::string(kGroupRanks)));

  if (distInfo_.backend.empty() && processGroupDesc == "default_pg") {
    distInfo_.backend = backend;
    distInfo_.rank = collectiveRecord.getMetadataValue(std::string(kRank));
    distInfo_.world_size = groupSize;
    // DistributedInfo carries only an NCCL version and there is no version
    // source for other backends, so populate it for NCCL and leave it empty
    // otherwise rather than mislabel non-NCCL traffic.
    if (backend == "nccl") {
      distInfo_.nccl_version = "unknown";
    }
  }

  auto pg_config = pgConfig();
  pg_config.pg_name = processGroupName;
  pg_config.pg_desc = std::move(processGroupDesc);
  pg_config.backend_config = backendConfig;
  pg_config.pg_size = groupSize;
  pg_config.ranks = std::move(groupRanks);
  pgMap_.insert({processGroupName, pg_config});
}

void ChromeTraceLogger::handleActivity(const libkineto::ITraceActivity& op) {
  if (!traceOf_) {
    return;
  }

  if (op.type() == ActivityType::CPU_INSTANT_EVENT) {
    handleGenericInstantEvent(op);
    return;
  }

  switch (op.type()) {
    case ActivityType::MTIA_COUNTERS:
    case ActivityType::XPU_SCOPE_PROFILER:
      handleCounterEvent(op);
      return;
    default:
      break;
  }

  int64_t ts = op.timestamp();
  int64_t duration = op.duration();

  duration = std::max<int64_t>(duration, 0);

  if (op.type() == ActivityType::GPU_USER_ANNOTATION) {
    // The GPU user annotations start at the same time as the
    // first associated GPU op. Since they appear later
    // in the trace file, this causes a visualization issue in Chrome.
    // Make it start one ns earlier and end 2 ns later.
    ts -= 1;
    duration += 2; // Still need it to end at the original point rounded up.
  }

  int external_id = 0;
  if (op.linkedActivity()) {
    external_id = op.linkedActivity()->correlationId();
  } else {
    // Some runtime events and kernels may not have a linked activity,
    // should not set an "External id" for them. Otherwise, these events
    // may be incorrectly linked to the other external events.
    static const std::set<libkineto::ActivityType> excludedTypes = {
        libkineto::ActivityType::GPU_MEMCPY,
        libkineto::ActivityType::GPU_MEMSET,
        libkineto::ActivityType::CONCURRENT_KERNEL,
        libkineto::ActivityType::CUDA_RUNTIME,
        libkineto::ActivityType::CUDA_DRIVER,
        libkineto::ActivityType::XPU_RUNTIME,
        libkineto::ActivityType::XPU_DRIVER,
        libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
        libkineto::ActivityType::PRIVATEUSE1_DRIVER};
    if (!excludedTypes.contains(op.type())) {
      external_id = op.correlationId();
    }
  }
  ArgsBuilder args;
  if (external_id != 0) {
    args.addRaw("External id", fmt::format("{}", external_id));
  }
  std::string op_metadata = op.metadataJson();
  sanitizeStrForJSON(op_metadata);
  args.appendFragment(op_metadata);

  // Populate collective metadata from the linked record_param_comms CPU op.
  const auto* linkedOp = op.linkedActivity();
  if (linkedOp != nullptr && linkedOp->name() == kParamCommsCallName) {
    if (op.type() == ActivityType::CONCURRENT_KERNEL) {
      appendCollectiveMetadata(args, *linkedOp, "nccl", "cuda:nccl");
    } else if (
        op.type() == ActivityType::MTIA_CCP_EVENTS &&
        op.name().starts_with("hccl::")) {
      appendCollectiveMetadata(args, *linkedOp, "hccl", "mtia:hccl");
    }
  }

  int64_t device = op.deviceId();
  int64_t resource = op.resourceId();

  // Move Stream Sync events to a dedicated row so they don't overlap with
  // kernel events on the same stream in the trace viewer.
  if (op.type() == ActivityType::CUDA_SYNC && op.name() == "Stream Sync") {
    int64_t syncTid = resource + kSyncStreamTidOffset;
    int64_t key = (device << 32) | syncTid;
    if (syncStreamMetadataEmitted_.insert(key).second) {
      int64_t metaTime = transToRelativeTime(ts);
      // clang-format off
      fmt::print(traceOf_, R"JSON(
  {{
    "name": "thread_name", "ph": "M", "ts": {}.{:03}, "pid": {}, "tid": {},
    "args": {{
      "name": "stream {} (sync)"
    }}
  }},
  {{
    "name": "thread_sort_index", "ph": "M", "ts": {}.{:03}, "pid": {}, "tid": {},
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
          metaTime/1000, metaTime%1000, device, syncTid,
          resource,
          metaTime/1000, metaTime%1000, device, syncTid,
          resource);
      // clang-format on
    }
    resource = syncTid;
  }

  // TODO: Remove this once legacy tools are updated.
  std::string op_name = op.name() == "kernel" ? "Kernel" : op.name();
  sanitizeStrForJSON(op_name);
  sanitizeForNonReadableChars(op_name);
  escapeQuotesForJSON(op_name);

  ts = transToRelativeTime(ts);
  writeCompleteEvent(
      /*cat=*/toString(op.type()),
      /*name=*/op_name,
      /*pid=*/device,
      /*tid=*/sanitizeTid(resource),
      /*ts=*/ts,
      /*dur=*/duration,
      /*args=*/args);
  if (op.flowId() > 0) {
    handleGenericLink(op);
  }
}

void ChromeTraceLogger::handleGenericActivity(
    const libkineto::GenericTraceActivity& op) {
  if (!traceOf_) {
    return;
  }
  handleActivity(op);
}

void ChromeTraceLogger::handleGenericLink(const ITraceActivity& act) {
  if (!traceOf_) {
    return;
  }
  static struct {
    int type;
    char name[16];
  } flow_names[] = {{kLinkFwdBwd, "fwdbwd"}, {kLinkAsyncCpuGpu, "ac2g"}};
  for (auto& flow : flow_names) {
    if (act.flowType() == flow.type) {
      // Link the activities via flow ID in source and destination.
      // The source node must return true from flowStart()
      // and the destination node false.
      if (act.flowStart()) {
        handleLink(kFlowStart, act, act.flowId(), flow.name);
      } else {
        handleLink(kFlowEnd, act, act.flowId(), flow.name);
      }
      return;
    }
  }
  LOG(WARNING) << "Unknown flow type: " << act.flowType();
}

void ChromeTraceLogger::handleLink(
    char type,
    const ITraceActivity& e,
    int64_t id,
    const std::string& name) {
  if (!traceOf_) {
    return;
  }

  int64_t ts = transToRelativeTime(e.timestamp());
  int64_t tid = e.resourceId();
  // Use the virtual tid for Stream Sync events to match the row
  // they were placed on in handleActivity().
  if (e.type() == ActivityType::CUDA_SYNC && e.name() == "Stream Sync") {
    tid = tid + kSyncStreamTidOffset;
  }
  writeFlowEvent(
      /*type=*/type,
      /*id=*/id,
      /*pid=*/e.deviceId(),
      /*tid=*/sanitizeTid(tid),
      /*ts=*/ts,
      /*cat=*/name,
      /*name=*/name);
}

void ChromeTraceLogger::finalizeTrace(
    [[maybe_unused]] const Config& config,
    [[maybe_unused]] std::unique_ptr<ActivityBuffers> buffers,
    int64_t endTime) {
  finalizeTrace(endTime);
}

void ChromeTraceLogger::addOnDemandDistMetadata() {
  if (distInfo_.backend.empty()) {
    return;
  }
  fmt::print(
      traceOf_,
      R"JSON(
  "distributedInfo": {{"backend": "{}", "rank": {}, "world_size": {}, "pg_count": {}, "pg_config": [)JSON",
      distInfo_.backend,
      distInfo_.rank,
      distInfo_.world_size,
      std::to_string(pgMap_.size()));

  bool first = true;
  for (const auto& element : pgMap_) {
    if (!first) {
      fmt::print(traceOf_, ",");
    }
    fmt::print(
        traceOf_,
        R"JSON({{"pg_name": "{}", "pg_desc": "{}", "backend_config": "{}", "pg_size": {}, "ranks": "{}"}})JSON",
        element.second.pg_name,
        element.second.pg_desc,
        element.second.backend_config,
        element.second.pg_size,
        element.second.ranks);
    first = false;
  }

  fmt::print(
      traceOf_,
      R"JSON(], "nccl_version": "{}"}},)JSON",
      distInfo_.nccl_version);
  distInfo_.distInfo_present_ = true;
}

void ChromeTraceLogger::finalizeTrace(int64_t endTime) {
  if (!traceOf_) {
    LOG(ERROR) << "Failed to write to log file!";
    return;
  }
  sanitizeStrForJSON(fileName_);
  LOG(INFO) << "Chrome Trace written to " << fileName_;

  // Note that this call ends the `traceEvents` array opened up in
  // `handleTraceStart.`
  endTime = transToRelativeTime(endTime);
  ArgsBuilder emptyArgs;
  writeInstantEvent(
      /*cat=*/"",
      /*name=*/"Record Window End",
      /*scope=*/"g",
      /*pid=*/R"("")",
      /*tid=*/R"("")",
      /*ts=*/endTime,
      /*args=*/emptyArgs,
      /*finalEvent=*/true);

  // Close the `traceEvents` array.
  fmt::print(traceOf_, "\n  ],");

  if (!distInfo_.distInfo_present_) {
    addOnDemandDistMetadata();
  }

  // The last entry MUST NOT end with a comma.
  fmt::print(traceOf_, R"JSON("traceName": "{}" }})JSON", fileName_);

  traceOf_.close();

  // On some systems, rename() fails if the destination file exists.
  // So, remove the destination file first.
  remove(fileName_.c_str());
  if (rename(tempFileName_.c_str(), fileName_.c_str()) != 0) {
    PLOG(ERROR) << "Failed to rename " << tempFileName_ << " to " << fileName_;
  } else {
    LOG(INFO) << "Renamed the trace file to " << fileName_;
  }
}

} // namespace KINETO_NAMESPACE
