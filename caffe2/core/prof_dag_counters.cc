#include "caffe2/core/prof_dag_counters.h"

#include <ostream>
#include <sstream>

namespace caffe2 {

ProfDAGCounters::ProfDAGCounters(const std::shared_ptr<const NetDef>& net_def) {
  report_.net_name_ = net_def->name();
  report_.num_runs_ = 0;
  auto num_ops = net_def->op_size();
  report_.op_types_.reserve(num_ops);
  for (auto op_id = 0; op_id < num_ops; ++op_id) {
    report_.op_types_.push_back(net_def->op(op_id).type());
  }
  report_.time_per_op_total_.resize(num_ops);
}

void ProfDAGCounters::ReportRunStart() {
  report_.num_runs_ += 1;
  timer_.Start();
  auto num_ops = report_.op_types_.size();
  op_start_times_run_.clear();
  op_start_times_run_.resize(num_ops, -1.0);
  op_end_times_run_.clear();
  op_end_times_run_.resize(num_ops, -1.0);
  op_async_end_times_run_.clear();
  op_async_end_times_run_.resize(num_ops, -1.0);
}

void ProfDAGCounters::AddPerOpStartTime(size_t op_id) {
  if (report_.num_runs_ <= 1) {
    return;
  }

  CAFFE_ENFORCE(op_id >= 0 && op_id < op_start_times_run_.size());
  op_start_times_run_[op_id] = timer_.MilliSeconds();
}

void ProfDAGCounters::AddPerOpEndTime(size_t op_id) {
  if (report_.num_runs_ <= 1) {
    return;
  }

  CAFFE_ENFORCE(op_id >= 0 && op_id < op_end_times_run_.size());
  op_end_times_run_[op_id] = timer_.MilliSeconds();
}

void ProfDAGCounters::AddPerOpAsyncEndTime(size_t op_id) {
  if (report_.num_runs_ <= 1) {
    return;
  }

  CAFFE_ENFORCE(op_id >= 0 && op_id < op_async_end_times_run_.size());
  op_async_end_times_run_[op_id] = timer_.MilliSeconds();
}

void ProfDAGCounters::ReportRunEnd() {
  if (report_.num_runs_ <= 1) {
    return;
  }

  auto runtime = timer_.MilliSeconds();
  report_.runtime_stats_ += ProfDAGStats(runtime);

  CaffeMap<std::string, float> cum_per_type_time_run_;
  CaffeMap<std::string, float> cum_per_type_invocations_run_;
  for (auto op_id = 0; op_id < report_.op_types_.size(); ++op_id) {
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

    report_.time_per_op_total_[op_id] += ProfDAGStats(op_time);

    const string& op_type = report_.op_types_[op_id];
    cum_per_type_time_run_[op_type] += op_time;
    cum_per_type_invocations_run_[op_type] += 1;
  }

  for (const auto& kv : cum_per_type_time_run_) {
    report_.time_per_op_type_total_[kv.first] += ProfDAGStats(kv.second);
    report_.times_per_run_per_type_total_[kv.first] +=
        ProfDAGStats(cum_per_type_invocations_run_[kv.first]);
  }
}

ProfDAGReport ProfDAGCounters::GetReport() const {
  return report_;
}

ProfDAGProto ProfDAGReport::statsProto(
    const std::string& name,
    const ProfDAGStats& stats) const {
  ProfDAGProto stats_proto;
  const auto& moments = stats.computeMoments();
  stats_proto.set_mean(moments.first);
  stats_proto.set_stddev(moments.second);
  stats_proto.set_name(name);
  return stats_proto;
}

ProfDAGProtos ProfDAGReport::GetOperatorStats() const {
  ProfDAGProtos prof_dag_protos;
  prof_dag_protos.set_net_name(net_name_);
  if (num_runs_ > 1) {
    for (auto& item : time_per_op_type_total_) {
      auto buf = prof_dag_protos.add_stats();
      buf->CopyFrom(statsProto(item.first, item.second));
    }
  }
  return prof_dag_protos;
}

ProfDAGProtos ProfDAGReport::GetPerOperatorCost() const {
  ProfDAGProtos prof_dag_protos;
  prof_dag_protos.set_net_name(net_name_);
  if (num_runs_ > 1) {
    for (int op_id = 0; op_id < op_types_.size(); op_id++) {
      const string& op_type = op_types_[op_id];
      auto buf = prof_dag_protos.add_stats();
      std::string op_output_name =
          net_name_ + "___" + to_string(op_id) + "___" + op_type;
      buf->CopyFrom(statsProto(op_output_name, time_per_op_total_[op_id]));
    }
  }
  return prof_dag_protos;
}

void ProfDAGReport::PrintStats() {
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

ProfDAGReport& ProfDAGReport::operator+=(const ProfDAGReport& rhs) {
  // Verify nets are compatible for addition
  CAFFE_ENFORCE_EQ(
      net_name_, rhs.net_name_, "Incompatible nets to add counters");
  CAFFE_ENFORCE_EQ(
      op_types_.size(),
      rhs.op_types_.size(),
      "Incompatible nets to add counters");
  for (auto idx = 0; idx < op_types_.size(); ++idx) {
    CAFFE_ENFORCE_EQ(
        op_types_[idx],
        rhs.op_types_[idx],
        "Incompatible nets to add counters");
  }

  if (rhs.num_runs_ <= 1) {
    // rhs does not have valid profiling results, do nothing
    return *this;
  } else if (num_runs_ <= 1) {
    // "this" does not have valid profiling results, but rhs does. copy rhs
    time_per_op_total_ = rhs.time_per_op_total_;
    time_per_op_type_total_ = rhs.time_per_op_type_total_;
    times_per_run_per_type_total_ = rhs.times_per_run_per_type_total_;
    runtime_stats_ = rhs.runtime_stats_;
    num_runs_ = rhs.num_runs_;
    return *this;
  }

  // Do the addition
  for (auto idx = 0; idx < time_per_op_total_.size(); ++idx) {
    time_per_op_total_[idx] += rhs.time_per_op_total_.at(idx);
  }
  for (auto& item : time_per_op_type_total_) {
    item.second += rhs.time_per_op_type_total_.at(item.first);
  }
  for (auto& item : times_per_run_per_type_total_) {
    item.second += rhs.times_per_run_per_type_total_.at(item.first);
  }
  runtime_stats_ += rhs.runtime_stats_;
  num_runs_ += rhs.num_runs_;

  return *this;
}

} // namespace caffe2
