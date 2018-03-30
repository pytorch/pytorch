#include "caffe2/core/net_simple.h"
#include "caffe2/core/net.h"

#include <iostream>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "caffe2/core/operator.h"
#include "caffe2/core/static_tracepoint.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

SimpleNet::SimpleNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : NetBase(net_def, ws) {
  VLOG(1) << "Constructing SimpleNet " << net_def->name();
  const bool net_def_has_device_option = net_def->has_device_option();
  // Initialize the operators
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    const auto& operator_def = net_def->op(idx);
    VLOG(1) << "Creating operator " << operator_def.name() << ": "
            << operator_def.type();
    std::unique_ptr<OperatorBase> op{nullptr};
    if (!operator_def.has_device_option() && net_def_has_device_option) {
      // In the case that the operator def does not specify a device option but
      // the net def has a default option, we copy the device option over to the
      // operator def.
      OperatorDef temp_def(operator_def);
      temp_def.mutable_device_option()->CopyFrom(net_def->device_option());
      op = CreateOperator(temp_def, ws, idx);
    } else {
      op = CreateOperator(operator_def, ws, idx);
      op->set_debug_def(
          std::shared_ptr<const OperatorDef>{net_def, &(net_def->op(idx))});
    }
    operators_.emplace_back(std::move(op));
  }
}

bool SimpleNet::Run() {
  StartAllObservers();
  VLOG(1) << "Running net " << name_;
  for (auto& op : operators_) {
    VLOG(1) << "Running operator " << op->debug_def().name() << "("
            << op->debug_def().type() << ").";
#ifdef CAFFE2_ENABLE_SDT
    const auto& op_name = op->debug_def().name().c_str();
    const auto& op_type = op->debug_def().type().c_str();
    auto* op_ptr = op.get();
    const auto& net_name = name_.c_str();
    CAFFE_SDT(operator_start, net_name, op_name, op_type, op_ptr);
#endif
    bool res = op->Run();
#ifdef CAFFE2_ENABLE_SDT
    CAFFE_SDT(operator_done, net_name, op_name, op_type, op_ptr);
#endif
    if (!res) {
      LOG(ERROR) << "Operator failed: " << ProtoDebugString(op->debug_def());
      return false;
    }
  }
  StopAllObservers();
  return true;
}

bool SimpleNet::RunAsync() {
  return Run();
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
  /* Use std::cout because logging may be disabled */
  std::cout << "Starting benchmark." << std::endl;
  std::cout << "Running warmup runs." << std::endl;
  CAFFE_ENFORCE(
      warmup_runs >= 0,
      "Number of warm up runs should be non negative, provided ",
      warmup_runs,
      ".");
  for (int i = 0; i < warmup_runs; ++i) {
    CAFFE_ENFORCE(Run(), "Warmup run ", i, " has failed.");
  }

  std::cout << "Main runs." << std::endl;
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
  std::cout << "Main run finished. Milliseconds per iter: "
            << millis / main_runs
            << ". Iters per second: " << 1000.0 * main_runs / millis << std::endl;

  vector<float> time_per_op(operators_.size(), 0);
  vector<uint64_t> flops_per_op;
  vector<uint64_t> memory_bytes_per_op;
  vector<uint64_t> param_bytes_per_op;
  CaffeMap<string, float> time_per_op_type;
  CaffeMap<string, float> flops_per_op_type;
  CaffeMap<string, float> memory_bytes_per_op_type;
  CaffeMap<string, float> param_bytes_per_op_type;
  if (run_individual) {
    for (int i = 0; i < main_runs; ++i) {
      for (auto& op : operators_) {
        op->ResetEvent();
      }
      int idx = 0;
      for (auto& op : operators_) {
        const string& op_type = op->debug_def().type();
        if (i == 0) { // Gather flops on the first run.
          auto* schema = OpSchemaRegistry::Schema(op_type);
          if (schema && schema->HasCostInferenceFunction()) {
            vector<TensorShape> shapes = op->InputTensorShapes();

            OpSchema::Cost cost = schema->InferCost(op->debug_def(), shapes);

            flops_per_op.emplace_back(cost.flops);
            memory_bytes_per_op.emplace_back(cost.bytes_moved);
            param_bytes_per_op.emplace_back(cost.params_bytes);

            flops_per_op_type[op_type] += cost.flops;
            memory_bytes_per_op_type[op_type] += cost.bytes_moved;
            param_bytes_per_op_type[op_type] += cost.params_bytes;
          }
        }
        timer.Start();
        CAFFE_ENFORCE(
            op->Run(),
            "operator ",
            op->debug_def().name(),
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
      const string& op_type = op->debug_def().type();
      const string& print_name =
          (op->debug_def().name().size()
               ? op->debug_def().name()
               : (op->debug_def().output_size() ? op->debug_def().output(0)
                                                : "NO_OUTPUT"));
      std::stringstream flops_str;
      if (idx < flops_per_op.size() && flops_per_op[idx]) {
        flops_str << " (" << to_string(1.0e-9 * flops_per_op[idx]) << " GFLOP, "
                  << to_string(1.0e-6 * flops_per_op[idx] / time_per_op[idx])
                  << " GFLOPS)";
      }
      std::stringstream memory_bytes_str;
      if (idx < memory_bytes_per_op.size() && memory_bytes_per_op[idx]) {
        memory_bytes_str << " (" << to_string(1.0e-6 * memory_bytes_per_op[idx])
                         << " MB)";
      }
      std::stringstream param_bytes_str;
      if (idx < param_bytes_per_op.size() && param_bytes_per_op[idx]) {
        memory_bytes_str << " (" << to_string(1.0e-6 * param_bytes_per_op[idx])
                         << " MB)";
      }
      std::cout << "Operator #" << idx << " (" << print_name << ", " << op_type
                << ") " << time_per_op[idx] / main_runs << " ms/iter"
                << flops_str.str() << memory_bytes_str.str()
                << param_bytes_str.str() << std::endl;
      ++idx;
    }
    const std::vector<string> metric(
        {"Time", "FLOP", "Feature Memory", "Parameter Memory"});
    const std::vector<double> normalizer(
        {1.0 / main_runs, 1.0e-9, 1.0e-6, 1.0e-6});
    const std::vector<string> unit({"ms", "GFLOP", "MB", "MB"});

    std::vector<CaffeMap<string, float>*> metric_per_op_type_vec_vec;
    metric_per_op_type_vec_vec.emplace_back(&time_per_op_type);
    metric_per_op_type_vec_vec.emplace_back(&flops_per_op_type);
    metric_per_op_type_vec_vec.emplace_back(&memory_bytes_per_op_type);
    metric_per_op_type_vec_vec.emplace_back(&param_bytes_per_op_type);
    for (int i = 0; i < metric_per_op_type_vec_vec.size(); ++i) {
      std::cout << metric[i] << " per operator type:" << std::endl;
      auto* item = metric_per_op_type_vec_vec[i];
      std::vector<std::pair<string, float>> metric_per_op_type_vec(
          (*item).begin(), (*item).end());
      std::sort(
          metric_per_op_type_vec.begin(),
          metric_per_op_type_vec.end(),
          PairLargerThan<string, float>);
      float total_metric = 0.;
      for (const auto& op_item : metric_per_op_type_vec) {
        total_metric += op_item.second * normalizer[i];
      }
      for (const auto& op_item : metric_per_op_type_vec) {
        float percent = 0.;
        if (total_metric > 0.) {
          percent = (100.0 * op_item.second * normalizer[i] / total_metric);
        }
        std::cout << std::setw(15) << std::setfill(' ')
                  << op_item.second * normalizer[i] << " " << unit[i] << ". "
                  << std::setw(10) << std::setfill(' ') << percent << "%. "
                  << op_item.first << std::endl;
      }
      std::cout << std::setw(15) << std::setfill(' ') << total_metric << " "
                << unit[i] << " in Total" << std::endl;
    }
  }
  // We will reuse time_per_op to return the result of BenchmarkNet.
  for (int i = 0; i < time_per_op.size(); ++i) {
    time_per_op[i] /= main_runs;
  }
  time_per_op.insert(time_per_op.begin(), millis / main_runs);
  return time_per_op;
}

REGISTER_NET(simple, SimpleNet);

} // namespace caffe2
