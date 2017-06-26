#include "caffe2/core/net.h"

#include <set>
#include <unordered_map>
#include <unordered_set>

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

CAFFE_DEFINE_REGISTRY(NetRegistry, NetBase, const NetDef&, Workspace*);

NetBase::NetBase(const NetDef& def, Workspace* /* unused */)
    : external_input_(def.external_input().begin(), def.external_input().end()),
      external_output_(
          def.external_output().begin(),
          def.external_output().end()),
      name_(def.name()) {
  // Go through the operators and make sure that blobs are correctly made.
  std::set<string> known_blobs(
      external_input_.begin(), external_input_.end());
  std::set<string> remaining_output(
      external_output_.begin(), external_output_.end());
  for (const auto& blob : known_blobs) {
    remaining_output.erase(blob);
  }
  for (const OperatorDef& op : def.op()) {
    for (const string& in : op.input()) {
      if (!known_blobs.count(in)) {
        if (external_input_.size()) {
          CAFFE_THROW(
              "op ",
              op.type(),
              ": Source for input ",
              in,
              " is unknown for net ",
              def.name(),
              ", operator ",
              ProtoDebugString(op));
        } else {
          // If we are not declaring input and output, we will simply VLOG it
          // for debugging purposes.
          VLOG(1) << "op " << op.type() << ": input " << in << " is unknown.";
        }
      }
    }
    for (const string& out : op.output()) {
      known_blobs.insert(out);
      remaining_output.erase(out);
    }
  }
  // Finally, check if all declared outputs are being created.
  CAFFE_ENFORCE(
      remaining_output.size() == 0,
      "Some of the blobs are declared as output but never produced by the "
      "net ",
      def.name(),
      ", the first one is ",
      *remaining_output.begin());
}

unique_ptr<NetBase> CreateNet(const NetDef& net_def, Workspace* ws) {
  // In default, we will return a simple network that just runs all operators
  // sequentially.
  if (!net_def.has_type()) {
    return make_unique<SimpleNet>(net_def, ws);
  }
  return NetRegistry()->Create(net_def.type(), net_def, ws);
}

SimpleNet::SimpleNet(const NetDef& net_def, Workspace* ws)
    : NetBase(net_def, ws) {
  VLOG(1) << "Constructing SimpleNet " << net_def.name();
  bool net_def_has_device_option = net_def.has_device_option();
  // Initialize the operators
  for (int idx = 0; idx < net_def.op_size(); ++idx) {
    const auto& operator_def = net_def.op(idx);
    VLOG(1) << "Creating operator " << operator_def.name()
            << ":" << operator_def.type();
    if (!operator_def.has_device_option() && net_def_has_device_option) {
      // In the case that the operator def does not specify a device option but
      // the net def has a default option, we copy the device option over to the
      // operator def.
      OperatorDef temp_def(operator_def);
      temp_def.mutable_device_option()->CopyFrom(net_def.device_option());
      operators_.emplace_back(CreateOperator(temp_def, ws, idx));
    } else {
      operators_.emplace_back(CreateOperator(operator_def, ws, idx));
    }
  }
}

bool SimpleNet::Run() {
  if (observer_) {
    observer_->Start();
  }
  VLOG(1) << "Running net " << name_;
  for (auto& op : operators_) {
    VLOG(1) << "Running operator " << op->def().name()
            << "(" << op->def().type() << ").";
    if (!op->Run()) {
      LOG(ERROR) << "Operator failed: "
                      << ProtoDebugString(op->def());
      return false;
    }
  }
  if (observer_) {
    observer_->Stop();
  }
  return true;
}

bool SimpleNet::RunAsync() {
  VLOG(1) << "Running net " << name_;
  for (auto& op : operators_) {
    VLOG(1) << "Running operator " << op->def().name()
            << "(" << op->def().type() << ").";
    if (!op->RunAsync()) {
      LOG(ERROR) << "Operator failed: "
                 << ProtoDebugString(op->def());
      return false;
    }
  }
  return true;
}

namespace {
template <typename A, typename B>
bool PairLargerThan(const std::pair<A, B>& x, const std::pair<A, B>& y) {
  return x.second > y.second;
}
}

vector<float> SimpleNet::TEST_Benchmark(
    const int warmup_runs,
    const int main_runs,
    const bool run_individual) {
  LOG(INFO) << "Starting benchmark.";
  LOG(INFO) << "Running warmup runs.";
  CAFFE_ENFORCE(
      warmup_runs >= 0,
      "Number of warm up runs should be non negative, provided ",
      warmup_runs,
      ".");
  for (int i = 0; i < warmup_runs; ++i) {
    CAFFE_ENFORCE(Run(), "Warmup run ", i, " has failed.");
  }

  LOG(INFO) << "Main runs.";
  CAFFE_ENFORCE(
      main_runs >= 0,
      "Number of main runs should be non negative, provided ",
      main_runs,
      ".");
  Timer timer;
  for (int i = 0; i < main_runs; ++i) {
    CAFFE_ENFORCE(Run(), "Main run ", i, " has failed.");
  }
  auto millis = timer.MilliSeconds();
  LOG(INFO) << "Main run finished. Milliseconds per iter: "
                 << millis / main_runs
                 << ". Iters per second: " << 1000.0 * main_runs / millis;

  vector<float> time_per_op(operators_.size(), 0);
  CaffeMap<string, float> time_per_op_type;
  if (run_individual) {
    for (int i = 0; i < main_runs; ++i) {
      int idx = 0;
      for (auto& op : operators_) {
        const string& op_type = op->def().type();
        timer.Start();
        CAFFE_ENFORCE(
            op->Run(),
            "operator ",
            op->def().name(),
            "(",
            op_type,
            ") has failed.");
        float spent = timer.MilliSeconds();
        time_per_op[idx] += spent;
        time_per_op_type[op_type] += spent;
        ++idx;
      }
    }

    int idx = 0;
    for (auto& op : operators_) {
      const string& op_type = op->def().type();
      const string& print_name =
          (op->def().name().size()
               ? op->def().name()
               : (op->def().output_size() ? op->def().output(0) : "NO_OUTPUT"));
      LOG(INFO) << "Operator #" << idx << " (" << print_name << ", " << op_type
                << ") " << time_per_op[idx] / main_runs << " ms/iter";
      ++idx;
    }
    LOG(INFO) << "Time per operator type:";
    // sort by decreasing time spending.
    std::vector<std::pair<string, float>> time_per_op_type_vec(
        time_per_op_type.begin(), time_per_op_type.end());
    std::sort(
        time_per_op_type_vec.begin(),
        time_per_op_type_vec.end(),
        PairLargerThan<string, float>);
    for (const auto& item : time_per_op_type_vec) {
      LOG(INFO) << std::setw(15) << std::setfill(' ') << item.second / main_runs
                << " " << item.first;
    }
  }
  // We will reuse time_per_op to return the result of BenchmarkNet.
  for (int i = 0; i < time_per_op.size(); ++i) {
    time_per_op[i] /= main_runs;
  }
  time_per_op.insert(time_per_op.begin(), millis / main_runs);
  return time_per_op;
}

namespace {

REGISTER_NET(simple, SimpleNet);

}

}  // namespace caffe2
