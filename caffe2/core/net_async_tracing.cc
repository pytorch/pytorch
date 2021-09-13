/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/core/net_async_tracing.h"

#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"

C10_DEFINE_string(
    caffe2_net_async_tracing_filepath,
    "/tmp",
    "Path to save tracing information");

C10_DEFINE_string(
    caffe2_net_async_names_to_trace,
    "",
    "Comma-separated list of net names to trace");

C10_DEFINE_int(caffe2_net_async_tracing_nth, 100, "Trace every Nth batch");

// For every Nth iterations, we will dump the tracing results to a json file
// The file is appended with the iteration number.
C10_DEFINE_int(
    caffe2_net_async_tracing_dumping_nth,
    10000,
    "Dump profiling result file every Nth batch");

namespace caffe2 {
namespace tracing {

int getCounterForNetName(const std::string& net_name) {
  // Append a unique number suffix because there could be multiple instances
  // of the same net and we want to uniquely associate each instance with
  // a profiling trace.
  static std::unordered_map<std::string, int> net_name_to_counter;
  static std::mutex map_mutex;
  std::unique_lock<std::mutex> map_lock(map_mutex);
  int counter = net_name_to_counter[net_name] + 1;
  net_name_to_counter[net_name] = counter;
  return counter;
}

Tracer::Tracer(
    const NetBase* net,
    const std::string& net_name,
    // NOLINTNEXTLINE(modernize-pass-by-value)
    TracingConfig config)
    : net_(net),
      filename_(net_name),
      iter_(0),
      dumping_iter_(0),
      config_(config) {
  std::replace(filename_.begin(), filename_.end(), '/', '_');
  filename_ = this->config().filepath + "/" + filename_ + "_id_" +
      c10::to_string(getCounterForNetName(net_name));
  timer_.Start();
}

void Tracer::recordEvent(const TracerEvent& event) {
  std::lock_guard<std::mutex> lock(tracer_mutex_);
  events_.push_back(event);
}

// Forward
int getUniqueShardId(const OperatorDef& op_def);

// Special handling of shard blob annotations
std::string Tracer::opTraceName(const OperatorBase* op) {
  int unique_shard_id =
      op->has_debug_def() ? getUniqueShardId(op->debug_def()) : -1;
  if (unique_shard_id != -1) {
    return op->type() + ":" + c10::to_string(unique_shard_id);
  } else {
    return op->type();
  }
}

std::string Tracer::opBlobsInfo(const OperatorBase& op) {
  std::string blobs_info;
  if (op.has_debug_def()) {
    blobs_info += "I: ";
    const auto& op_def = op.debug_def();
    for (const auto& input : op_def.input()) {
      blobs_info += input + "; ";
    }
    blobs_info += "O: ";
    for (const auto& output : op_def.output()) {
      blobs_info += output + "; ";
    }
  }
  return blobs_info;
}

std::string Tracer::serializeEvent(const TracerEvent& event) {
  std::stringstream serialized_event;
  serialized_event << std::fixed;
  serialized_event << "{\n";
  serialized_event << " \"ts\": " << event.timestamp_ << ",\n";
  serialized_event << " \"pid\": 0,\n"; // not using pid field
  if (event.thread_label_ >= 0) {
    serialized_event << " \"tid\": " << event.thread_label_ << ",\n";
  } else {
    serialized_event << " \"tid\": " << event.tid_ << ",\n";
  }

  if (event.is_beginning_) {
    std::unordered_map<std::string, int> int_args;
    std::unordered_map<std::string, std::string> string_args;
    if (event.name_) {
      // NOLINTNEXTLINE(modernize-raw-string-literal)
      serialized_event << " \"name\": \"" << event.name_ << "\",\n";
    } else if (event.op_id_ >= 0) {
      auto* op = net_->GetOperators().at(event.op_id_);
      // NOLINTNEXTLINE(modernize-raw-string-literal)
      serialized_event << " \"name\": \"" << opTraceName(op) << "\",\n";
    } else {
      serialized_event << " \"name\": \"n/a\",\n";
    }

    if (event.category_) {
      // NOLINTNEXTLINE(modernize-raw-string-literal)
      serialized_event << " \"cat\": \"" << event.category_ << "\",\n";
    } else {
      serialized_event << " \"cat\": \"net\",\n";
    }

    if (event.op_id_ >= 0) {
      auto* op = net_->GetOperators().at(event.op_id_);
      int_args["op_id"] = event.op_id_;
      int_args["device_type"] = op->device_option().device_type();
      int_args["device_id"] = DeviceId(op->device_option());
      string_args["blobs"] = opBlobsInfo(*op);
    }

    if (event.task_id_ >= 0) {
      int_args["task_id"] = event.task_id_;
    }

    if (event.iter_ >= 0) {
      int_args["iter_id"] = event.iter_;
    }

    if (event.stream_id_ >= 0) {
      int_args["stream_id"] = event.stream_id_;
    }

    // NOLINTNEXTLINE(modernize-raw-string-literal)
    serialized_event << " \"ph\": \"B\"";
    if (!int_args.empty() || !string_args.empty()) {
      serialized_event << ",\n \"args\": {\n";
      auto left_to_output = int_args.size() + string_args.size();
      for (const auto& kv : int_args) {
        serialized_event << "  \"" << kv.first << "\": " << kv.second;
        --left_to_output;
        if (left_to_output > 0) {
          serialized_event << ",\n";
        }
      }
      for (const auto& kv : string_args) {
        serialized_event << "  \"" << kv.first << "\": \"" << kv.second << "\"";
        --left_to_output;
        if (left_to_output > 0) {
          serialized_event << ",\n";
        }
      }
      serialized_event << "\n }";
    }
  } else {
    serialized_event << " \"ph\": \"E\"\n";
  }
  serialized_event << "\n}";

  return serialized_event.str();
}

// fix occasional cases with zero duration events
void Tracer::linearizeEvents() {
  std::unordered_map<long, long> time_offsets;
  std::unordered_map<long, long> last_times;
  std::hash<std::thread::id> hasher;
  const long time_eps = 1; // us
  for (auto& event : events_) {
    long tid =
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        (event.thread_label_ >= 0) ? event.thread_label_ : hasher(event.tid_);
    auto event_ts = event.timestamp_;
    if (last_times.count(tid)) {
      event_ts += time_offsets[tid];
      CAFFE_ENFORCE(event_ts >= last_times[tid]);
      if (event_ts <= last_times[tid] + time_eps) {
        event_ts += time_eps;
        time_offsets[tid] += time_eps;
      } else if (event_ts > last_times[tid] + 2 * time_eps) {
        long eps_len = (event_ts - last_times[tid]) / time_eps;
        if (time_offsets[tid] >= time_eps * (eps_len - 1)) {
          time_offsets[tid] -= time_eps * (eps_len - 1);
          event_ts -= time_eps * (eps_len - 1);
        } else {
          event_ts -= time_offsets[tid];
          time_offsets[tid] = 0;
        }
      }
      event.timestamp_ = event_ts;
      last_times[tid] = event_ts;
    } else {
      last_times[tid] = event_ts;
      time_offsets[tid] = 0;
    }
  }
}

void Tracer::renameThreads() {
  std::unordered_map<long, int> tids;
  std::unordered_map<int, int> numa_counters;
  std::unordered_map<long, int> tid_to_numa;
  std::hash<std::thread::id> hasher;
  const long numa_multiplier = 1000000000;
  for (auto& event : events_) {
    if (event.thread_label_ >= 0 || event.op_id_ < 0) {
      continue;
    }
    auto* op = net_->GetOperators().at(event.op_id_);
    if (!op->device_option().has_numa_node_id()) {
      continue;
    }
    int numa_node_id = op->device_option().numa_node_id();
    CAFFE_ENFORCE_GE(numa_node_id, 0, "Invalid NUMA node id: ", numa_node_id);
    long tid = hasher(event.tid_);

    if (!tid_to_numa.count(tid)) {
      tid_to_numa[tid] = numa_node_id;
    } else {
      CAFFE_ENFORCE_EQ(tid_to_numa[tid], numa_node_id);
    }

    if (!numa_counters.count(numa_node_id)) {
      numa_counters[numa_node_id] = 1;
    }
    if (!tids.count(tid)) {
      tids[tid] = numa_counters[numa_node_id]++;
    }
    event.thread_label_ = numa_multiplier * (numa_node_id + 1) + tids[tid];
  }
}

void Tracer::setEnabled(bool enabled) {
  enabled_ = enabled;
}

bool Tracer::isEnabled() const {
  return enabled_;
}

int Tracer::bumpIter() {
  return iter_++;
}

int Tracer::getIter() {
  return iter_;
}

int Tracer::bumpDumpingIter() {
  return dumping_iter_++;
}

void Tracer::dumpTracingResultAndClearEvents(const std::string& file_suffix) {
  if (events_.empty() || filename_.empty()) {
    return;
  }
  linearizeEvents();
  renameThreads();
  std::stringstream serialized;
  serialized << "[\n";
  for (size_t idx = 0; idx < events_.size(); ++idx) {
    serialized << serializeEvent(events_[idx]);
    if (idx != events_.size() - 1) {
      serialized << ",\n";
    }
  }
  serialized << "\n]\n";

  auto output_file_name = filename_ + "_iter_" + file_suffix + ".json";
  LOG(INFO) << "Dumping profiling result file to " << output_file_name;
  WriteStringToFile(serialized.str(), output_file_name.c_str());
  events_.clear();
}

Tracer::~Tracer() {
  dumpTracingResultAndClearEvents("final_batch");
}

thread_local TracerGuard* current_tracer_guard;

void TracerGuard::init(Tracer* tracer) {
  enabled_ = tracer && tracer->isEnabled();
  if (enabled_) {
    current_tracer_guard = this;
  }
  tracer_ = tracer;
}

void TracerGuard::addArgument() {}

void TracerGuard::addArgument(TracingField field, const char* value) {
  switch (field) {
    case TRACE_NAME: {
      event_.name_ = value;
      break;
    }
    case TRACE_CATEGORY: {
      event_.category_ = value;
      break;
    }
    default: {
      CAFFE_THROW("Unexpected tracing string field ", field);
    }
  }
}

void TracerGuard::addArgument(TracingField field, int value) {
  switch (field) {
    case TRACE_OP: {
      event_.op_id_ = value;
      break;
    }
    case TRACE_TASK: {
      event_.task_id_ = value;
      break;
    }
    case TRACE_STREAM: {
      event_.stream_id_ = value;
      break;
    }
    case TRACE_THREAD: {
      event_.thread_label_ = value;
      break;
    }
    case TRACE_ITER: {
      event_.iter_ = value;
      break;
    }
    default: {
      CAFFE_THROW("Unexpected tracing int field ", field);
    }
  }
}

void TracerGuard::recordEventStart() {
  if (enabled_) {
    if (event_.thread_label_ < 0) {
      event_.tid_ = std::this_thread::get_id();
    }
    event_.is_beginning_ = true;
    event_.timestamp_ = (long)caffe2::round(tracer_->timer_.MicroSeconds());
    tracer_->recordEvent(event_);
  }
}

TracerGuard::~TracerGuard() {
  if (enabled_) {
    event_.is_beginning_ = false;
    event_.timestamp_ = (long)caffe2::round(tracer_->timer_.MicroSeconds());
    tracer_->recordEvent(event_);
    if (current_tracer_guard == this) {
      current_tracer_guard = nullptr;
    }
  }
}

void TracerGuard::disable() {
  enabled_ = false;
}

TracerGuard* TracerGuard::getCurrentTracerGuard() {
  return current_tracer_guard;
}

int extractShardId(const std::string& name) {
  const std::string kShard = "shard:";
  // We sometimes have multiple shards, but actually need the last one, hence
  // using rfind here. Hacky but it works till we pass shard id in graph
  // metadata.
  auto pos = name.rfind(kShard);
  if (pos != std::string::npos) {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    int left_pos = pos + kShard.length();
    int right_pos = left_pos;
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    while (right_pos < name.length() && isdigit(name[right_pos])) {
      right_pos++;
    }
    return c10::stoi(name.substr(left_pos, right_pos - left_pos));
  } else {
    return -1;
  }
}

// Return unique shard id, or -1 if it is not unique.
int getUniqueShardId(const OperatorDef& op_def) {
  int unique_shard_id = -1;
  for (const auto& names : {op_def.input(), op_def.output()}) {
    for (const auto& name : names) {
      int shard_id = extractShardId(name);
      if (shard_id != -1) {
        if (unique_shard_id != -1) {
          return -1;
        }
        unique_shard_id = shard_id;
      }
    }
  }
  return unique_shard_id;
}

bool isTraceableNetName(const std::string& net_name) {
  auto tracing_nets = caffe2::split(',', FLAGS_caffe2_net_async_names_to_trace);
  return !net_name.empty() &&
      std::find(tracing_nets.begin(), tracing_nets.end(), net_name) !=
      tracing_nets.end();
}

bool hasEnableTracingFlag(const NetBase* net) {
  if (!net->has_debug_def()) {
    return false;
  }
  return GetFlagArgument(net->debug_def(), "enable_tracing", false);
}

TracingConfig getTracingConfigFromNet(const NetBase* net) {
  ArgumentHelper arg_helper(net->debug_def());
  TracingConfig cfg;

  cfg.mode = (arg_helper.GetSingleArgument<std::string>("tracing_mode", "") ==
              "GLOBAL_TIMESLICE")
      ? TracingMode::GLOBAL_TIMESLICE
      : TracingMode::EVERY_K_ITERATIONS;

  cfg.filepath = arg_helper.GetSingleArgument<std::string>(
      "tracing_filepath", FLAGS_caffe2_net_async_tracing_filepath);

  cfg.trace_every_nth_batch = arg_helper.GetSingleArgument<int>(
      "trace_every_nth_batch", FLAGS_caffe2_net_async_tracing_nth);
  cfg.dump_every_nth_batch = arg_helper.GetSingleArgument<int>(
      "dump_every_nth_batch", FLAGS_caffe2_net_async_tracing_dumping_nth);

  cfg.trace_for_n_ms =
      arg_helper.GetSingleArgument<int>("trace_for_n_ms", cfg.trace_for_n_ms);
  cfg.trace_every_n_ms = arg_helper.GetSingleArgument<int>(
      "trace_every_n_ms", cfg.trace_every_n_ms);

  return cfg;
};

std::shared_ptr<Tracer> create(
    const NetBase* net,
    const std::string& net_name) {
  // Enable the tracer if the net has the "enable_tracing" argument set OR
  // if the command line option includes the net name option in the list of
  // traceable nets.
  bool trace_net = hasEnableTracingFlag(net) || isTraceableNetName(net_name);
  return trace_net
      ? std::make_shared<Tracer>(net, net_name, getTracingConfigFromNet(net))
      : nullptr;
}

bool startIter(const std::shared_ptr<Tracer>& tracer) {
  if (!tracer) {
    return false;
  }
  auto iter = tracer->bumpIter();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool is_enabled;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool should_dump;
  if (tracer->config().mode == TracingMode::EVERY_K_ITERATIONS) {
    is_enabled = iter % tracer->config().trace_every_nth_batch == 0;
    should_dump = iter % tracer->config().dump_every_nth_batch == 0;
  } else {
    using namespace std::chrono;
    auto ms =
        duration_cast<milliseconds>(system_clock::now().time_since_epoch())
            .count();
    is_enabled = (ms % tracer->config().trace_every_n_ms) <
        tracer->config().trace_for_n_ms;
    // dump just after disabled tracing
    should_dump = tracer->isEnabled() && !is_enabled;
  }
  tracer->setEnabled(is_enabled);
  if (should_dump) {
    int dumping_iter = tracer->bumpDumpingIter();
    tracer->dumpTracingResultAndClearEvents(c10::to_string(dumping_iter));
  }
  return is_enabled;
}

} // namespace tracing

} // namespace caffe2
