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

#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"
#include "nnapi.h"

namespace caffe2 {

namespace {

static double benchmark_conv_caffe2(
    Workspace* ws,
    int N,
    int C,
    int H,
    int W,
    int K,
    int kernel,
    int group,
    int warmup = 5,
    int run = 10,
    std::string engine = "NNPACK") {
  caffe2::Workspace localWs;
  if (!ws) {
    ws = &localWs;
  }
  {
    auto* t = BlobGetMutableTensor(ws->CreateBlob("X_cpu"), CPU);
    t->Resize(N, C, H, W);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(
        t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }
  {
    auto* t = BlobGetMutableTensor(ws->CreateBlob("W"), CPU);
    if (group == 1) {
      t->Resize(K, C, kernel, kernel);
    } else {
      t->Resize(K, 1, kernel, kernel);
    }
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(
        t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }
  {
    auto* t = BlobGetMutableTensor(ws->CreateBlob("B"), CPU);
    t->Resize(K);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(
        t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }

  OperatorDef op;
  {
    op.set_type("Conv");
    op.add_input("X_cpu");
    op.add_input("W");
    op.add_input("B");
    op.add_output("Y_cpu");
    op.set_engine(engine);
    {
      auto& arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NCHW");
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("convolution_transform_strategy");
      arg.set_s("PRECOMPUTE");
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("kernel");
      arg.set_i(kernel);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("group");
      arg.set_i(group);
    }
  }

  // NNPack
  std::unique_ptr<caffe2::OperatorBase> op1(CreateOperator(op, ws));

  Timer timer;
  CAFFE_ENFORCE(op1->Run());
  for (int i = 0; i < warmup; i++) {
    op1->Run();
  }
  timer.Start();
  for (int i = 0; i < run; i++) {
    op1->Run();
  }
  return double(timer.MilliSeconds()) / run;
}

static double benchmark_conv_nnapi(
    Workspace* ws,
    int N,
    int C,
    int H,
    int W,
    int K,
    int kernel,
    int group,
    int warmup = 5,
    int run = 10) {
  caffe2::Workspace localWs;
  if (!ws) {
    ws = &localWs;
  }
  {
    auto* t = BlobGetMutableTensor(ws->CreateBlob("X_cpu"), CPU);
    t->Resize(N, H, W, C);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(
        t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }
  {
    auto* t = BlobGetMutableTensor(ws->CreateBlob("W"), CPU);
    if (group > 1) {
      CAFFE_ENFORCE_EQ(C, group);
      t->Resize(1, kernel, kernel, C);
    } else {
      t->Resize(K, kernel, kernel, C);
    }
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(
        t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }
  {
    auto* t = BlobGetMutableTensor(ws->CreateBlob("B"), CPU);
    t->Resize(K);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(
        t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }

  NetDef netdef;
  {
    {
      auto& op = *(netdef.add_op());
      op.set_type("Conv");
      op.add_input("X_cpu");
      op.add_input("W");
      op.add_input("B");
      op.add_output("Y_cpu");
      {
        auto& arg = *(op.add_arg());
        arg.set_name("order");
        arg.set_s("NHWC");
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("kernel");
        arg.set_i(kernel);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("group");
        arg.set_i(group);
      }
    }
    netdef.add_external_input("X_cpu");
    netdef.add_external_input("W");
    netdef.add_external_input("B");
    netdef.add_external_output("Y_cpu");
  }

  // NN API
  NetDef initNet;
  NNApi model(initNet, netdef, ws);
  std::vector<TensorCPU*> inputs, outputs;
  inputs.push_back(BlobGetMutableTensor(ws->GetBlob("X_cpu"), CPU));
  CAFFE_ENFORCE(model.run(inputs, &outputs));

  for (int i = 0; i < warmup; i++) {
    model.run(inputs, &outputs);
  }
  Timer timer;
  timer.Start();
  for (int i = 0; i < run; i++) {
    model.run(inputs, &outputs);
  }
  return double(timer.MilliSeconds()) / run;
}

static double benchmark_conv_nnapi_int8(
    Workspace* ws,
    int N,
    int C,
    int H,
    int W,
    int K,
    int kernel,
    int group,
    int warmup = 5,
    int run = 10) {
  caffe2::Workspace localWs;
  if (!ws) {
    ws = &localWs;
  }
  {
    auto* t = BlobGetMutableTensor(ws->CreateBlob("X_cpu"), CPU);
    t->Resize(N, H, W, C);
    for (int i = 0; i < t->size(); i++) {
      t->mutable_data<uint8_t>()[i] = rand() % 10;
    }
  }
  {
    auto* t = BlobGetMutableTensor(ws->CreateBlob("W"), CPU);
    if (group > 1) {
      CAFFE_ENFORCE_EQ(C, group);
      t->Resize(1, kernel, kernel, C);
    } else {
      t->Resize(K, kernel, kernel, C);
    }
    for (int i = 0; i < t->size(); i++) {
      t->mutable_data<uint8_t>()[i] = rand() % 10;
    }
  }

  // For input tensor of ANEURALNETWORKS_TENSOR_QUANT8_ASYMM type, the bias
  // should be of ANEURALNETWORKS_TENSOR_INT32, with zeroPoint of 0 and
  // bias_scale == input_scale * filter_scale.
  {
    auto* t = BlobGetMutableTensor(ws->CreateBlob("B"), CPU);
    t->Resize(K);
    for (int i = 0; i < t->size(); i++) {
      t->mutable_data<int32_t>()[i] = rand() % 10;
    }
  }

  NetDef netdef;
  {
    {
      auto& op = *(netdef.add_op());
      op.set_type("Conv");
      op.add_input("X_cpu");
      op.add_input("W");
      op.add_input("B");
      op.add_output("Y_cpu");
      {
        auto& arg = *(op.add_arg());
        arg.set_name("order");
        arg.set_s("NHWC");
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("kernel");
        arg.set_i(kernel);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("group");
        arg.set_i(group);
      }
      // Hack
      // for weight tensor
      {
        auto& arg = *(op.add_arg());
        arg.set_name("weight_scale");
        arg.set_f(1.0);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("weight_zero_point");
        arg.set_i(0);
      }
      // for output tensor
      // For output tensor of ANEURALNETWORKS_TENSOR_QUANT8_ASYMM type, the
      // following condition must be satisfied: output_scale > input_scale *
      // filter_scale
      {
        auto& arg = *(op.add_arg());
        arg.set_name("output_scale");
        arg.set_f(2.0);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("output_zero_point");
        arg.set_i(0);
      }
    }
    netdef.add_external_input("X_cpu");
    netdef.add_external_input("W");
    netdef.add_external_input("B");
    netdef.add_external_output("Y_cpu");
    // scale and zero_point for the input tensor
    {
      auto& arg = *(netdef.add_arg());
      arg.set_name("scale");
      arg.set_f(1.0);
    }
    {
      auto& arg = *(netdef.add_arg());
      arg.set_name("zero_point");
      arg.set_i(0);
    }
  }

  // NN API
  NetDef initNet;
  NNApi model(initNet, netdef, ws);
  std::vector<TensorCPU*> inputs, outputs;
  inputs.push_back(BlobGetMutableTensor(ws->GetBlob("X_cpu"), CPU));
  CAFFE_ENFORCE(model.run(inputs, &outputs));

  for (int i = 0; i < warmup; i++) {
    model.run(inputs, &outputs);
  }
  Timer timer;
  timer.Start();
  for (int i = 0; i < run; i++) {
    model.run(inputs, &outputs);
  }
  return double(timer.MilliSeconds()) / run;
}

} // namespace

} // namespace caffe2

int main(int argc, char** argv) {
  caffe2::Workspace ws;
  ws.GetThreadPool()->setMinWorkSize(0);

  int warmup = 2, mainrun = 10;
  // float32
  for (int space : {14, 26, 52, 104}) {
    for (int input_channel : {64, 128, 256, 512}) {
      for (int kernel : {1, 3}) {
        int output_channel = input_channel;
        const double cpu_time = caffe2::benchmark_conv_caffe2(
            &ws,
            1,
            input_channel,
            space,
            space,
            output_channel,
            kernel,
            1,
            warmup,
            mainrun,
            "NNPACK");
        const double nn_time_fp32 = caffe2::benchmark_conv_nnapi(
            &ws,
            1,
            input_channel,
            space,
            space,
            output_channel,
            kernel,
            1,
            warmup,
            mainrun);
        const double nn_time_int8 = caffe2::benchmark_conv_nnapi_int8(
            &ws,
            1,
            input_channel,
            space,
            space,
            output_channel,
            kernel,
            1,
            warmup,
            mainrun);
        const double flops = double(input_channel) * output_channel * kernel *
            kernel * (kernel == 1 ? space : space - 2) *
            (kernel == 1 ? space : space - 2) * 2;
        printf(
            "Conv: X: %ix%i  \tC: %i -> %i\tK: %ix%i\t32b"
            "NNPACK GFLOPS: %.2f\t32b"
            "NN-API GFLOPS: %.2f\t8b"
            "NN-API GOPS: %.2f\n",
            space,
            space,
            input_channel,
            output_channel,
            kernel,
            kernel,
            flops / cpu_time / 1E6,
            flops / nn_time_fp32 / 1E6,
            flops / nn_time_int8 / 1E6);
      }
    }
  }
  fflush(stdout);

  // depthwise
  for (int space : {14, 26, 52, 104}) {
    for (int channel : {64, 128, 256, 512}) {
      for (int kernel : {3}) {
        const double cpu_time = caffe2::benchmark_conv_caffe2(
            &ws,
            1,
            channel,
            space,
            space,
            channel,
            kernel,
            channel,
            warmup,
            mainrun,
            "DEPTHWISE_3x3");
        const double nn_time_fp32_dwise = caffe2::benchmark_conv_nnapi(
            &ws,
            1,
            channel,
            space,
            space,
            channel,
            kernel,
            channel,
            warmup,
            mainrun);
        const double nn_time_int8_dwise = caffe2::benchmark_conv_nnapi_int8(
            &ws,
            1,
            channel,
            space,
            space,
            channel,
            kernel,
            channel,
            warmup,
            mainrun);
        const double dwise_bandwidth = sizeof(float) * double(channel) *
            (space * space + kernel == 1
                 ? space * space
                 : (space - 2) * (space - 2) + kernel * kernel);
        printf(
            "Conv: X: %ix%i  \tC: %i -> %i\tK: %ix%i\t32b"
            "Caffe2 Dwise GB/s: %.2f\t32b"
            "NN-API Dwise GB/s: %.2f\t8b"
            "NN-API Dwise GB/s: %.2f\n",
            space,
            space,
            channel,
            channel,
            kernel,
            kernel,
            dwise_bandwidth / cpu_time / 1E6,
            dwise_bandwidth / nn_time_fp32_dwise / 1E6,
            dwise_bandwidth / sizeof(float) / nn_time_int8_dwise / 1E6);
      }
    }
  }
}
