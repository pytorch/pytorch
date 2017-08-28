#ifdef __ARM_NEON__
#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/proto_utils.h"

#define TEST_REAL_DATA 0
// If you want to test with real data you may want to grab this
// script P57273314 and a 227x227 png of a cat or something.
#if TEST_REAL_DATA
#include "data_chw.h"
#include "data_hwc.h"
#define POPULATE_DATA(_n, _s, _l) do {\
  Blob* _blob = ws.CreateBlob((_n));\
  auto* _tensor = _blob->GetMutable<TensorCPU>();\
  _tensor->Resize((_s));\
  memcpy(_tensor->mutable_data<float>(), data_##_l, _tensor->nbytes());\
} while(0)
#else
// Rough test on static data
#define POPULATE_DATA(_n, _s, _l) do {\
  Blob* _blob = ws.CreateBlob((_n));\
  auto* _tensor = _blob->GetMutable<TensorCPU>();\
  _tensor->Resize((_s));\
  memset(_tensor->mutable_data<float>(), 1, _tensor->nbytes());\
} while(0)
#endif

#include <cmath>
#include <random>
#include <iostream>
#include <fstream>

namespace caffe2 {

void AddConstInput(const vector<TIndex>& shape,
                   const float value,
                   const string& name,
                   Workspace* ws) {
  DeviceOption option;
  CPUContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = blob->GetMutable<TensorCPU>();
  tensor->Resize(shape);
  math::Set<float, CPUContext>(tensor->size(), value,
                               tensor->mutable_data<float>(),
                               &context);
}

void AddNoiseInput(const vector<TIndex>& shape,
                   const string& name,
                   Workspace* ws) {
  DeviceOption option;
  CPUContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = blob->GetMutable<TensorCPU>();
  tensor->Resize(shape);

  math::RandGaussian<float, CPUContext>(
    tensor->size(),
    0.0f, 10.0f,
    tensor->mutable_data<float>(),
    &context);
}


float snpe_run(int iters, Workspace& ws) {
  const int H = 227;
  const int W = 227;
  const int C = 3;

  POPULATE_DATA("X_snpe", (caffe2::vector<caffe2::TIndex>{H, W, C}), hwc);
  
  OperatorDef def;
  def.set_name("snpe_test");
  def.set_type("SNPE");
  def.add_input("X_snpe");
  def.add_output("snpeout");
  std::ostringstream model_buffer;
  std::ifstream file("/data/local/tmp/squeeze_net.dlc", std::ios::in|std::ios::binary);
  CAFFE_ENFORCE(file.is_open(), "Couldn't open test model.");
  model_buffer << file.rdbuf();
  CAFFE_ENFORCE(model_buffer.str().length() > 0, "Couldn't load model into string.");
  def.add_arg()->CopyFrom(MakeArgument("model_buffer", model_buffer.str()));

  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  assert(op.get());
  Timer timer;
  timer.Start();
  for (auto i = 0; i < iters; ++i) {
    op->Run();
  }
  return timer.MicroSeconds();
}

float caffe2_run(int iters, Workspace& ws) {
  NetDef init_net;
  NetDef predict_net;

  const int N = 1;
  const int H = 227;
  const int W = 227;
  const int C = 3;

  ReadProtoFromBinaryFile("/data/local/tmp/squeeze_init_net.pb", &init_net);
  ReadProtoFromBinaryFile("/data/local/tmp/squeeze_predict_net.pb", &predict_net);
  ws.RunNetOnce(init_net);
  POPULATE_DATA("data", (caffe2::vector<caffe2::TIndex>{N, C, H, W}), chw);
  predict_net.set_name("SqueezeNet");
  ws.CreateNet(predict_net);

  // Timing caffe2
  Timer timer;
  timer.Start();
  for (auto i = 0; i < iters; ++i) {
    ws.RunNet("SqueezeNet");
  }
  float us = timer.MicroSeconds();

  OperatorDef copy_def;
  copy_def.set_type("Copy");
  copy_def.set_name("Copy");
  copy_def.add_input("softmaxout");
  copy_def.add_output("caffe2out");
  unique_ptr<OperatorBase> copy_op(CreateOperator(copy_def, &ws));
  copy_op->Run();
  return us;
}

} // caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::Workspace ws;
  int iters = 50;

  std::cout << "Testing caffe2...";
  float t_caffe2 = caffe2::caffe2_run(iters, ws);
  std::cout << "done!\nTesting snpe...";
  float t_snpe = caffe2::snpe_run(iters, ws);
  std::cout << "done!\n";

  caffe2::Blob* caffe2_out_blob = ws.GetBlob("caffe2out");
  auto& caffe2_tensor = caffe2_out_blob->Get<caffe2::TensorCPU>();
  caffe2::Blob* snpe_out_blob = ws.GetBlob("snpeout");
  auto& snpe_tensor = snpe_out_blob->Get<caffe2::TensorCPU>();

  CAFFE_ENFORCE(snpe_tensor.size() == caffe2_tensor.size(), "Outputs are not the same!\n");

  float total_diff = 0;
  float KL_divergence = 0;
  float JS_divergence = 0;
  float max = 0;
  int max_index = 0;

  for (auto i = 0; i < snpe_tensor.size(); ++i) {
    auto Q = caffe2_tensor.data<float>()[i];
    auto P = snpe_tensor.data<float>()[i];
    if (Q > max) {
      max = Q;
      max_index = i;
    }
    auto diff = fabs(P - Q);
    auto avg = P + Q / 2;
    if (P && Q) {
      KL_divergence += P * log(P / Q);
      JS_divergence += 0.5 * P * log(P / Q) + 0.5 * Q * log(Q / P);
    }
    total_diff += diff;
    if (diff / avg > 0.10 && avg > 0.01) { // 10% difference and a non trivial confidence
      std::cout << "Diff: " << diff << " (" << P << " vs " << Q << ")\n";
    }
  }

  float avg_diff = total_diff; // Avg difference as percentage (not a great metric)
  printf("Average difference is %f%%\n", avg_diff * 100);
  printf("JS Divergence is %f\n", JS_divergence); // Jensen-Shannon
  printf("KL Divergence is %f\n", KL_divergence); // Kullback–Leibler
  printf("Predicted %d with %f%% confidence\n", max_index, max * 100);

  printf ("Caffe2: %f microseconds.\n", t_caffe2);
  printf ("SNPE: %f microseconds.\n", t_snpe);
  printf ("SNPE impl %fx faster\n", t_caffe2/t_snpe);
  return 0;
}
#else
// Compile for different targets.
int main() {
  return 0;
}
#endif
