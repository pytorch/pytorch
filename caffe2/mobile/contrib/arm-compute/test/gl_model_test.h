#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include "caffe2/mobile/contrib/arm-compute/test/gl_operator_test.h"
#include "caffe2/mobile/contrib/arm-compute/core/rewrite_net.h"
#include <gtest/gtest.h>

#include "caffe2/core/operator.h"
#include "caffe2/core/workspace.h"
#include <unordered_set>

CAFFE2_DEFINE_int(warmup, 10, "The number of iterations to warm up.");
CAFFE2_DEFINE_int(iter, 100, "The number of iterations to run.");
CAFFE2_DEFINE_bool(
    run_individual,
    false,
    "Whether to benchmark individual operators.");


constexpr float tol = 0.03;
namespace caffe2 {
  void benchmarkModel(std::string init_net_pb, std::string predict_net_pb, std::string input_name, std::vector<int> input_dims, std::string net_name="benchmark_net", std::unordered_set<std::string> cpu_ops = std::unordered_set<std::string>({})) {
    unique_ptr<caffe2::Workspace> ws(new caffe2::Workspace());
    NetDef init_net_def;
    CAFFE_ENFORCE(ReadProtoFromFile(init_net_pb, &init_net_def));
    CAFFE_ENFORCE(ws->RunNetOnce(init_net_def));
    NetDef predict_net_def, predict_net_def_gpu;
    CAFFE_ENFORCE(ReadProtoFromFile(predict_net_pb, &predict_net_def));
    PopulateCPUBlob(ws.get(), true, input_name, input_dims);

    tryConvertToOpenGL(predict_net_def, &predict_net_def_gpu, false, cpu_ops);
    // change the name of last op
    auto index = predict_net_def_gpu.op().size() - 1;
    auto last_blob = predict_net_def_gpu.op()[index].output()[0];
    auto op = predict_net_def_gpu.mutable_op(index);
    auto output = op->mutable_output(0);
    *output = last_blob + "_gpu";

    for (auto i = 0; i < predict_net_def_gpu.external_output_size(); ++i) {
      auto out = predict_net_def_gpu.mutable_external_output(i);
      if (*out == last_blob) {
        *out = last_blob + "_gpu";
      }
    }

    compareNetResult4D(*ws, predict_net_def, predict_net_def_gpu, last_blob, last_blob + "_gpu");

  NetBase* net = ws->CreateNet(predict_net_def);
  LOG(INFO) << "[C2DEBUG] Benchmarking OpenGL Net";
  net->TEST_Benchmark(caffe2::FLAGS_warmup, caffe2::FLAGS_iter, caffe2::FLAGS_run_individual);
  // Test CPU
  for (auto i = 0; i < predict_net_def.op().size(); ++i) {
    auto op = predict_net_def.mutable_op(i);
    if (std::find(cpu_ops.begin(), cpu_ops.end(), op->type()) == cpu_ops.end()) {
      op->mutable_device_option()->set_device_type(CPU);
    }
  }
  predict_net_def.set_type("simple");
  predict_net_def.set_name("cpu_net");
  net = ws->CreateNet(predict_net_def);
  LOG(INFO) << "[C2DEBUG] Benchmarking CPU Net";
  net->TEST_Benchmark(caffe2::FLAGS_warmup, caffe2::FLAGS_iter, caffe2::FLAGS_run_individual);

  }
} // namespace caffe2
