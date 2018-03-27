#include "caffe2/mobile/contrib/arm-compute/core/net_gl.h"
#include "caffe2/mobile/contrib/arm-compute/core/context.h"
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

GLNet::GLNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : NetBase(net_def, ws) {
  ws_ = ws;
  VLOG(1) << "Constructing GLNet " << net_def->name();
  const bool net_def_has_device_option = net_def->has_device_option();
  // Initialize the operators
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    const auto& operator_def = net_def->op(idx);
    VLOG(1) << "Creating operator " << operator_def.name() << ": "
            << operator_def.type();
    output_blobs_.push_back(operator_def.output(0));
    if (operator_def.has_device_option() && operator_def.device_option().device_type() == OPENGL) {
      opengl_device_.push_back(true);
    } else {
      opengl_device_.push_back(false);
    }

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

bool GLNet::Run() {
  StartAllObservers();
  if (first_run_) {
    first_run_ = false;
    for (auto& op: operators_) {
      if (op->device_option().device_type() == OPENGL) {
        op->Run();
      }
    }
  }
  VLOG(1) << "Running net " << name_;
  for (auto& op : operators_) {
    bool res = op->Run();
    if (!res) {
      LOG(ERROR) << "Operator failed: " << ProtoDebugString(op->debug_def());
      return false;
    }
  }
  StopAllObservers();
  return true;
}

bool GLNet::RunAsync() {
  return Run();
}

namespace {
template <typename A, typename B>
bool PairLargerThan(const std::pair<A, B>& x, const std::pair<A, B>& y) {
  return x.second > y.second;
}
}

vector<float> GLNet::TEST_Benchmark(
    const int warmup_runs,
    const int main_runs,
    const bool run_individual) {
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

  auto last_blob = output_blobs_[output_blobs_.size() - 1];
  Blob *gpu_out_blob = ws_->GetBlob(last_blob);
  auto &g_ = gpu_out_blob->Get<GLTensor<half>>();
  // Enforce gpu execution
  g_.sync();

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
  g_.sync();

  auto millis = timer.MilliSeconds();
  std::cout << "[C2DEBUG] Main run finished. Milliseconds per iter: "
            << millis / main_runs
            << ". Iters per second: " << 1000.0 * main_runs / millis << std::endl;

  vector<float> time_per_op(operators_.size(), 0);
  vector<uint64_t> flops_per_op(operators_.size(), 0);
  CaffeMap<string, float> time_per_op_type;
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
            flops_per_op[idx] =
                schema->InferCost(op->debug_def(), shapes).flops;
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
        if (opengl_device_[idx]) {
          Blob *gpu_out_blob = ws_->GetBlob(output_blobs_[idx]);
          auto &g_ = gpu_out_blob->Get<GLTensor<half>>();
          g_.sync();
        }
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
      if (flops_per_op[idx]) {
        flops_str << " ("
                  << to_string(1.0e-6 * flops_per_op[idx] / time_per_op[idx])
                  << " GFLOPS)";
      }
      std::cout << "[C2DEBUG] Operator #" << idx << " (" << print_name << ", " << op_type
                << ") " << time_per_op[idx] / main_runs << " ms/iter"
                << flops_str.str() << std::endl;
      ++idx;
    }
    std::cout << "[C2DEBUG] Time per operator type:" << std::endl;
    // sort by decreasing time spending.
    std::vector<std::pair<string, float>> time_per_op_type_vec(
        time_per_op_type.begin(), time_per_op_type.end());
    std::sort(
        time_per_op_type_vec.begin(),
        time_per_op_type_vec.end(),
        PairLargerThan<string, float>);
    for (const auto& item : time_per_op_type_vec) {
      std::cout << "[C2DEBUG] " << std::setw(15) << std::setfill(' ') << item.second / main_runs
                << " " << item.first << std::endl;
    }
  }
  // We will reuse time_per_op to return the result of BenchmarkNet.
  for (int i = 0; i < time_per_op.size(); ++i) {
    time_per_op[i] /= main_runs;
  }
  time_per_op.insert(time_per_op.begin(), millis / main_runs);
  return time_per_op;
}

REGISTER_NET(opengl, GLNet);

} // namespace caffe2
