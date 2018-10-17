#include "caffe2/core/prof_dag_counters.h"

#include <ostream>
#include <sstream>

namespace caffe2 {

ProfDAGCounters::ProfDAGCounters(const std::shared_ptr<const NetDef>& net_def)
    : net_name_(net_def->name()), num_runs_(0) {
  op_types_.reserve(net_def->op_size());
  for (auto op_id = 0; op_id < net_def->op_size(); ++op_id) {
    op_types_.push_back(net_def->op(op_id).type());
  }
  time_per_op_total_.resize(op_types_.size());
}

void ProfDAGCounters::ReportRunStart() {
  num_runs_ += 1;
  timer_.Start();

  op_start_times_run_.clear();
  op_start_times_run_.resize(op_types_.size(), -1.0);
  op_end_times_run_.clear();
  op_end_times_run_.resize(op_types_.size(), -1.0);
  op_async_end_times_run_.clear();
  op_async_end_times_run_.resize(op_types_.size(), -1.0);
}

void ProfDAGCounters::AddPerOpStartTime(size_t op_id) {
  if (num_runs_ <= 1) {
    return;
  }

  CAFFE_ENFORCE(op_id >= 0 && op_id < op_start_times_run_.size());
  op_start_times_run_[op_id] = timer_.MilliSeconds();
}

void ProfDAGCounters::AddPerOpEndTime(size_t op_id) {
  if (num_runs_ <= 1) {
    return;
  }

  CAFFE_ENFORCE(op_id >= 0 && op_id < op_end_times_run_.size());
  op_end_times_run_[op_id] = timer_.MilliSeconds();
}

void ProfDAGCounters::AddPerOpAsyncEndTime(size_t op_id) {
  if (num_runs_ <= 1) {
    return;
  }

  CAFFE_ENFORCE(op_id >= 0 && op_id < op_async_end_times_run_.size());
  op_async_end_times_run_[op_id] = timer_.MilliSeconds();
}

void ProfDAGCounters::ReportRunEnd() {
  if (num_runs_ <= 1) {
    return;
  }

  auto runtime = timer_.MilliSeconds();
  runtime_stats_ += ProfDAGStats(runtime);

  CaffeMap<std::string, float> cum_per_type_time_run_;
  CaffeMap<std::string, float> cum_per_type_invocations_run_;
  for (auto op_id = 0; op_id < op_types_.size(); ++op_id) {
    float op_time;
    CAFFE_ENFORCE(op_start_times_run_[op_id] > 0);
    if (op_async_end_times_run_[op_id] > 0) {
      auto op_async_time =
          op_async_end_times_run_[op_id] - op_start_times_run_[op_id];
      CAFFE_ENFORCE_GE(op_async_time, 0.0);
      op_time = op_async_time;
    } else {
      auto op_sync_time = op_end_times_run_[op_id] - op_start_times_run_[op_id];
      CAFFE_ENFORCE_GE(op_sync_time, 0.0);
      op_time = op_sync_time;
    }

    time_per_op_total_[op_id] += ProfDAGStats(op_time);

    const string& op_type = op_types_[op_id];
    cum_per_type_time_run_[op_type] += op_time;
    cum_per_type_invocations_run_[op_type] += 1;
  }

  for (const auto& kv : cum_per_type_time_run_) {
    time_per_op_type_total_[kv.first] += ProfDAGStats(kv.second);
    times_per_run_per_type_total_[kv.first] +=
        ProfDAGStats(cum_per_type_invocations_run_[kv.first]);
  }
}

ProfDAGProto ProfDAGCounters::statsProto(
    const std::string& name,
    const ProfDAGStats& stats) const {
  ProfDAGProto stats_proto;
  const auto& moments = stats.computeMoments();
  stats_proto.set_mean(moments.first);
  stats_proto.set_stddev(moments.second);
  stats_proto.set_name(name);
  return stats_proto;
}

ProfDAGProtos ProfDAGCounters::GetOperatorStats() const {
  CAFFE_ENFORCE_GT(num_runs_, 1, "Insufficient number of runs");
  ProfDAGProtos prof_dag_protos;
  for (auto& item : time_per_op_type_total_) {
    auto buf = prof_dag_protos.add_stats();
    buf->CopyFrom(statsProto(item.first, item.second));
  }
  return prof_dag_protos;
}

ProfDAGProtos ProfDAGCounters::GetPerOperatorCost() const {
  CAFFE_ENFORCE_GT(num_runs_, 1, "Insufficient number of runs");
  ProfDAGProtos prof_dag_protos;
  for (int op_id = 0; op_id < op_types_.size(); op_id++) {
    const string& op_type = op_types_[op_id];
    auto buf = prof_dag_protos.add_stats();
    std::string op_output_name =
        net_name_ + "___" + to_string(op_id) + "___" + op_type;
    buf->CopyFrom(statsProto(op_output_name, time_per_op_total_[op_id]));
  }
  return prof_dag_protos;
}

void ProfDAGCounters::PrintStats() {
  if (num_runs_ <= 1) {
    LOG(INFO) << "Insufficient number of runs";
    return;
  }

  std::ostringstream debug_out;
  debug_out << "Measured operators over " << num_runs_ << " net runs ("
            << net_name_ << "), #ops: " << op_types_.size() << std::endl;

  debug_out << "Mean time in operator type per run (stddev):" << std::endl;
  for (const auto& item : time_per_op_type_total_) {
    const auto& moments = item.second.computeMoments();
    const auto& times_moments =
        times_per_run_per_type_total_[item.first].computeMoments();
    debug_out << std::setw(10) << std::setfill(' ') << moments.first
              << " ms/run (" << std::setw(10) << std::setfill(' ')
              << moments.second << " ms/run) "
              << " Op count per run: " << times_moments.first << "  "
              << item.first << std::endl;
  }
  const auto& runtime_moments = runtime_stats_.computeMoments();
  debug_out << net_name_ << " runtime: " << runtime_moments.first << " ms ("
            << runtime_moments.second << " ms)" << std::endl;

  LOG(INFO) << debug_out.str();
}

} // namespace caffe2
