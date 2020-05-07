#include "caffe2/core/prof_dag_counters.h"
#include "caffe2/utils/string_utils.h"

#include <ostream>
#include <sstream>

namespace caffe2 {

ProfDAGCounters::ProfDAGCounters(const std::shared_ptr<const NetDef>& net_def) {
  report_.net_name_ = net_def->name();
  report_.num_runs_ = 0;
  auto num_ops = net_def->op_size();
  report_.op_types_.reserve(num_ops);
  report_.op_extra_info_.reserve(num_ops);

  for (auto op_id = 0; op_id < num_ops; ++op_id) {
    const auto& op = net_def->op(op_id);
    if (op.engine() == "") {
      report_.op_types_.push_back(op.type());
    } else {
      report_.op_types_.push_back(op.type() + "(" + op.engine() + ")");
    }
    vector<std::string> op_extra_info;
    if (op.has_device_option() && op.device_option().extra_info_size() > 0) {
      for (auto i = 0; i < op.device_option().extra_info_size(); ++i) {
        std::string extra_info_str = op.device_option().extra_info(i);
        op_extra_info.push_back(extra_info_str);
      }
    }
    report_.op_extra_info_.push_back(op_extra_info);
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

  CaffeMap<std::string, float> cum_per_type_time_run;
  CaffeMap<std::string, float> cum_per_type_invocations_run;
  std::vector<float> per_op_time_run(report_.op_types_.size(), 0.0);
  for (auto op_id = 0U; op_id < report_.op_types_.size(); ++op_id) {
    // check that we have valid times, otherwise return;
    // times might not be valid if network execution ended prematurely
    // because of operator errors
    if (op_start_times_run_[op_id] < 0.0) {
      return;
    }

    float op_time = 0.0;
    if (op_async_end_times_run_[op_id] > 0.0) {
      op_time = op_async_end_times_run_[op_id] - op_start_times_run_[op_id];
    } else {
      if (op_end_times_run_[op_id] < 0.0) {
        return;
      }
      op_time = op_end_times_run_[op_id] - op_start_times_run_[op_id];
    }

    per_op_time_run[op_id] = op_time;

    const string& op_type = report_.op_types_[op_id];
    cum_per_type_time_run[op_type] += op_time;
    cum_per_type_invocations_run[op_type] += 1;
  }

  // all operator times are valid, update report stats
  report_.runtime_stats_ += ProfDAGStats(runtime);

  for (auto op_id = 0U; op_id < report_.op_types_.size(); ++op_id) {
    report_.time_per_op_total_[op_id] += ProfDAGStats(per_op_time_run[op_id]);
  }

  for (const auto& kv : cum_per_type_time_run) {
    report_.time_per_op_type_total_[kv.first] += ProfDAGStats(kv.second);
    report_.times_per_run_per_type_total_[kv.first] +=
        ProfDAGStats(cum_per_type_invocations_run[kv.first]);
  }
}

ProfDAGReport ProfDAGCounters::GetReport() const {
  return report_;
}

bool ProfDAGReport::hasStats() const {
  return runtime_stats_.cnt() > 0;
}

ProfDAGProto ProfDAGReport::statsProto(
    const std::string& name,
    const ProfDAGStats& stats,
    const std::vector<std::string>& op_extra_info) const {
  ProfDAGProto stats_proto;
  const auto& moments = stats.computeMoments();
  stats_proto.set_mean(moments.first);
  stats_proto.set_stddev(moments.second);
  stats_proto.set_name(name);
  for (auto& extra_info : op_extra_info) {
    stats_proto.add_extra_info(extra_info);
  }
  return stats_proto;
}

ProfDAGProtos ProfDAGReport::GetOperatorStats() const {
  ProfDAGProtos prof_dag_protos;
  prof_dag_protos.set_net_name(net_name_);
  if (hasStats()) {
    for (auto& item : time_per_op_type_total_) {
      auto buf = prof_dag_protos.add_stats();
      buf->CopyFrom(statsProto(item.first, item.second, vector<std::string>()));
    }
  }
  return prof_dag_protos;
}

ProfDAGProtos ProfDAGReport::GetPerOperatorCost() const {
  ProfDAGProtos prof_dag_protos;
  prof_dag_protos.set_net_name(net_name_);
  if (hasStats()) {
    for (auto op_id = 0U; op_id < op_types_.size(); op_id++) {
      const string& op_type = op_types_[op_id];
      auto buf = prof_dag_protos.add_stats();
      std::string op_output_name =
          net_name_ + "___" + to_string(op_id) + "___" + op_type;
      buf->CopyFrom(statsProto(
          op_output_name, time_per_op_total_[op_id], op_extra_info_[op_id]));
    }
  }
  return prof_dag_protos;
}

void ProfDAGReport::PrintStats() {
  if (!hasStats()) {
    LOG(INFO) << "Insufficient number of runs";
    return;
  }

  std::ostringstream debug_out;
  debug_out << "Measured operators over " << runtime_stats_.cnt()
            << " net runs (" << net_name_ << "), #ops: " << op_types_.size()
            << std::endl;

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
  for (auto idx = 0U; idx < op_types_.size(); ++idx) {
    CAFFE_ENFORCE_EQ(
        op_types_[idx],
        rhs.op_types_[idx],
        "Incompatible nets to add counters");
  }

  if (!rhs.hasStats()) {
    // rhs does not have valid profiling results, do nothing
    return *this;
  } else if (!hasStats()) {
    // "this" does not have valid profiling results, but rhs does. copy rhs
    time_per_op_total_ = rhs.time_per_op_total_;
    time_per_op_type_total_ = rhs.time_per_op_type_total_;
    times_per_run_per_type_total_ = rhs.times_per_run_per_type_total_;
    runtime_stats_ = rhs.runtime_stats_;
    num_runs_ = rhs.num_runs_;
    return *this;
  }

  // Do the addition
  for (auto idx = 0U; idx < time_per_op_total_.size(); ++idx) {
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
