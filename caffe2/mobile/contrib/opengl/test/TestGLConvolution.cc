// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/core/workspace.h"

#include "../core/GL.h"
#include "../core/GLLogging.h"
#include "../core/arm_neon_support.h"
#include "../operators/gl_tiling_utils.h"
#include "TestGLConvolution.h"
#include "opengl_test.h"

#include <vector>

void AddNoiseInput(const std::vector<caffe2::TIndex>& shape,
                   const std::string& name,
                   caffe2::Workspace* ws) {
  caffe2::CPUContext context;
  caffe2::Blob* blob = ws->CreateBlob(name);
  auto* tensor = blob->GetMutable<caffe2::TensorCPU>();
  tensor->Resize(shape);

  caffe2::math::RandGaussian<float, caffe2::CPUContext>(
      tensor->size(), 0.0f, 10.0f, tensor->mutable_data<float>(), &context);
}

double BenchOp(const std::string& typ,
               int inputC,
               int outputC,
               int kW,
               int kH,
               int stride,
               int inW,
               int inH,
               bool transposed,
               caffe2::Workspace* ws = nullptr) {
  caffe2::Workspace localWs;
  if (!ws) {
    ws = &localWs;
  }

  const char* engine = transposed ? "MOBILE" : "NNPACK";

  caffe2::OperatorDef def1;
  def1.set_name("test");
  def1.set_type(typ);
  def1.set_engine(engine);
  def1.add_input("X");
  def1.add_input("W");
  def1.add_input("B");
  def1.add_output("Y");

  def1.add_arg()->CopyFrom(caffe2::MakeArgument("kernel_h", kH));
  def1.add_arg()->CopyFrom(caffe2::MakeArgument("kernel_w", kW));
  def1.add_arg()->CopyFrom(caffe2::MakeArgument("stride_h", stride));
  def1.add_arg()->CopyFrom(caffe2::MakeArgument("stride_w", stride));
  def1.add_arg()->CopyFrom(caffe2::MakeArgument("pad_t", 0));
  def1.add_arg()->CopyFrom(caffe2::MakeArgument("pad_l", 0));
  def1.add_arg()->CopyFrom(caffe2::MakeArgument("pad_b", 0));
  def1.add_arg()->CopyFrom(caffe2::MakeArgument("pad_r", 0));
  def1.add_arg()->CopyFrom(
      caffe2::MakeArgument("convolution_transform_strategy", std::string("PRECOMPUTE")));

  AddNoiseInput(std::vector<caffe2::TIndex>{1, inputC, inH, inW}, "X", ws);
  if (transposed) {
    AddNoiseInput(std::vector<caffe2::TIndex>{inputC, outputC, kH, kW}, "W", ws);
  } else {
    AddNoiseInput(std::vector<caffe2::TIndex>{outputC, inputC, kH, kW}, "W", ws);
  }
  AddNoiseInput(std::vector<caffe2::TIndex>{outputC}, "B", ws);

  std::unique_ptr<caffe2::OperatorBase> op1(CreateOperator(def1, ws));

  // Measure one iteration
  caffe2::Timer timer;
  timer.Start();

  op1->Run();

  float one_iteration = timer.MilliSeconds();

  int target_iterations = std::max((int)(1000 / one_iteration), 1);
  int warmup_iterations = std::max((int)(200 / one_iteration), 1);

  // warm up
  for (int i = 0; i < warmup_iterations; i++) {
    op1->Run();
  }

  timer.Start();

  int runs = target_iterations;
  for (int i = 0; i < runs; i++) {
    op1->Run();
  }

  auto total_t = timer.MilliSeconds();

  gl_log(GL_LOG,
         "%s(%d -> %d, %dx%d - %dx%d - %s) took: %.4f ms/iter\n",
         typ.c_str(),
         inputC,
         outputC,
         inW,
         inH,
         kW,
         kH,
         engine,
         timer.MilliSeconds() / (float)runs);
  return double(total_t) / runs;
}

template <typename T>
static double BenchGLConvolution(int input_channels,
                                 int output_channels,
                                 int kernel_width,
                                 int kernel_height,
                                 int input_width,
                                 int input_height,
                                 int input_padding,
                                 int input_stride,
                                 bool transposed,
                                 caffe2::Workspace* ws = nullptr) {
  int tile_x = 1, tile_y = 1;
  caffe2::squareFactors((input_channels + 3) / 4, tile_x, tile_y);

  gl_log(GL_LOG, "Input Tiles Factors: %d, %d\n", tile_x, tile_y);

  caffe2::Workspace localWs;
  if (!ws) {
    ws = &localWs;
  }

  AddNoiseInput(
      std::vector<caffe2::TIndex>{1, input_channels, input_height, input_width}, "X_cpu", ws);
  if (transposed) {
    AddNoiseInput(
        std::vector<caffe2::TIndex>{input_channels, output_channels, kernel_height, kernel_width},
        "W",
        ws);
  } else {
    AddNoiseInput(
        std::vector<caffe2::TIndex>{output_channels, input_channels, kernel_height, kernel_width},
        "W",
        ws);
  }
  AddNoiseInput(std::vector<caffe2::TIndex>{output_channels}, "b", ws);

  caffe2::NetDef netdef;
  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu");
    op.add_output("X_gl");
    {
      auto& arg = *(op.add_arg());
      arg.set_name("tile_x");
      arg.set_i(tile_x);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("tile_y");
      arg.set_i(tile_y);
    }
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type(transposed ? "OpenGLConvTranspose" : "OpenGLConv");
    op.add_input("X_gl");
    {
      op.add_input("W");
      op.add_input("b");
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NCHW");
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("kernel");
      arg.set_i(kernel_height);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("pad");
      arg.set_i(input_padding);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("stride");
      arg.set_i(input_stride);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("is_last");
      arg.set_i(1);
    }
    op.add_output("Y_gl");
  }

  std::vector<std::unique_ptr<caffe2::OperatorBase>> ops;

  for (auto& op : netdef.op()) {
    ops.push_back(CreateOperator(op, ws));
  }

  // Run the Copy Operator
  ops[0]->Run();

  // Make sure the tested operator is precompiled
  ops[1]->Run();
  glFinish();

  // Measure one iteration
  caffe2::Timer timer;
  timer.Start();

  ops[1]->Run();
  glFinish();

  float one_iteration = timer.MilliSeconds();

  int target_iterations = std::max((int)(1000 / one_iteration), 1);
  int warmup_iterations = std::max((int)(200 / one_iteration), 1);

  // warm up
  for (int i = 0; i < warmup_iterations; i++) {
    ops[1]->Run();
  }
  glFinish();

  timer.Start();

  int runs = target_iterations;
  for (int i = 0; i < runs; i++) {
    ops[1]->Run();
  }
  glFinish();

  const double gpuIterTime = double(timer.MilliSeconds()) / runs;

  gl_log(GL_LOG,
         "%s(%d -> %d, %dx%d - %dx%d - OpenGL) took: %.4f ms/iter\n",
         transposed ? "ConvTranspose" : "Conv",
         input_channels,
         output_channels,
         input_width,
         input_height,
         kernel_width,
         kernel_height,
         gpuIterTime);

  return gpuIterTime;
}

void TestSegmentationUseCase(caffe2::Workspace& ws) {
  int stride = 1;
  bool transposed = false;

  std::vector<int> kernels({3});
  std::vector<int> sizes({192, 96, 48, 24, 12, 6, 6, 12, 24, 48, 96, 192});
  std::vector<std::pair<int, int>> channels({{3, 12},
                                             {12, 24},
                                             {24, 48},
                                             {48, 96},
                                             {96, 180},
                                             {180, 220},
                                             {220, 180},
                                             {180, 96},
                                             {96, 24},
                                             {48, 24},
                                             {24, 12},
                                             {12, 1}});

  for (int i = 0; i < sizes.capacity(); i++) {
    int input_channels = channels[i].first;
    int output_channels = channels[i].second;
    int size = sizes[i];
    for (const auto& kernel : kernels) {
      int tile_x = 1, tile_y = 1;
      caffe2::squareFactors((input_channels + 3) / 4, tile_x, tile_y);

      caffe2::testOpenGLConv(1,
                             input_channels,
                             size,
                             size,
                             output_channels,
                             3,
                             3,
                             0,
                             1,
                             caffe2::Conv,
                             0.1 * input_channels / 8,
                             true,
                             1,
                             1,
                             tile_x,
                             tile_y,
                             true);

      const double gpuIterTime = BenchGLConvolution<float16_t>(
          input_channels, output_channels, kernel, kernel, size, size, 0, stride, transposed, &ws);
      const double cpuIterTime = BenchOp(transposed ? "ConvTranspose" : "Conv",
                                         input_channels,
                                         output_channels,
                                         kernel,
                                         kernel,
                                         stride,
                                         size,
                                         size,
                                         transposed,
                                         &ws);
      const double flops = double(input_channels) * output_channels * kernel * kernel *
                           (kernel == 1 ? size : size - 2) * (kernel == 1 ? size : size - 2) * 2;
      printf(
          "Conv: X: %ix%i  \tC: %i -> %i\tK: %ix%i\t16b GPU GFLOPS: %.2f\t32b CPU GFLOPS:"
          "%.2f\tratio: "
          "%.2f\n",
          size,
          size,
          input_channels,
          output_channels,
          kernel,
          kernel,
          flops / gpuIterTime / 1E6,
          flops / cpuIterTime / 1E6,
          cpuIterTime / gpuIterTime);
    }
  }
}

void TestsConvolutionRange(caffe2::Workspace& ws) {
  printf("running convolution benchmarks\n");

  bool transposed = false;
  int stride = 1;

  std::vector<int> kernels({1, 3});
  std::vector<int> sizes({1080, 720, 312, 104, 52, 26, 14});
  std::vector<int> channels({4, 8, 16, 32, 64, 128, 256});

  for (int i = 0; i < sizes.capacity(); i++) {
    int input_channels = channels[i];
    int output_channels = input_channels;
    int size = sizes[i];
    for (const auto& kernel : kernels) {
      int tile_x = 1, tile_y = 1;
      caffe2::squareFactors((input_channels + 3) / 4, tile_x, tile_y);

      caffe2::testOpenGLConv(1,
                             input_channels,
                             size,
                             size,
                             output_channels,
                             3,
                             3,
                             0,
                             1,
                             caffe2::Conv,
                             0.1 * input_channels / 8,
                             true,
                             1,
                             1,
                             tile_x,
                             tile_y,
                             true);

      const double gpuIterTime = BenchGLConvolution<float16_t>(input_channels,
                                                               output_channels,
                                                               kernel,
                                                               kernel,
                                                               size,
                                                               size,
                                                               0,
                                                               stride,
                                                               transposed,
                                                               &ws);
      const double cpuIterTime = BenchOp(transposed ? "ConvTranspose" : "Conv",
                                         input_channels,
                                         output_channels,
                                         kernel,
                                         kernel,
                                         stride,
                                         size,
                                         size,
                                         transposed,
                                         &ws);
      const double flops = double(input_channels) * output_channels * kernel * kernel *
      (kernel == 1 ? size : size - 2) *
      (kernel == 1 ? size : size - 2) * 2;
      printf(
             "Conv: X: %ix%i  \tC: %i -> %i\tK: %ix%i\t16b GPU GFLOPS: %.2f\t32b CPU GFLOPS:"
             "%.2f\tratio: "
             "%.2f\n",
             size,
             size,
             input_channels,
             output_channels,
             kernel,
             kernel,
             flops / gpuIterTime / 1E6,
             flops / cpuIterTime / 1E6,
             cpuIterTime / gpuIterTime);
    }
  }
}

void TestGLConvolution() {
  caffe2::Workspace ws;
  ws.GetThreadPool()->setMinWorkSize(0);

  TestSegmentationUseCase(ws);

  TestsConvolutionRange(ws);

  //  // ConvTranspose
  //  BenchGLConvolution<float16_t>(16, 16, 3, 3, 640, 360, 0, 2, true);
  //  BenchGLConvolution<float16_t>(16, 16, 4, 4, 640, 360, 0, 2, true);
  //  BenchGLConvolution<float16_t>(16, 16, 5, 5, 640, 360, 0, 2, true);
  //  BenchGLConvolution<float16_t>(16, 16, 6, 6, 640, 360, 0, 2, true);
  //  BenchGLConvolution<float16_t>(16, 16, 7, 7, 640, 360, 0, 2, true);
  //  BenchGLConvolution<float16_t>(16, 16, 8, 8, 640, 360, 0, 2, true);
  //  BenchGLConvolution<float16_t>(16, 16, 9, 9, 640, 360, 0, 2, true);
  //
  //  BenchOp("ConvTranspose", 16, 16, 3, 3, 2, 640, 360, true);
  //  BenchOp("ConvTranspose", 16, 16, 4, 4, 2, 640, 360, true);
  //  BenchOp("ConvTranspose", 16, 16, 5, 5, 2, 640, 360, true);
  //  BenchOp("ConvTranspose", 16, 16, 6, 6, 2, 640, 360, true);
  //  BenchOp("ConvTranspose", 16, 16, 7, 7, 2, 640, 360, true);
  //  BenchOp("ConvTranspose", 16, 16, 8, 8, 2, 640, 360, true);
  //  BenchOp("ConvTranspose", 16, 16, 9, 9, 2, 640, 360, true);
  //
  //  // Conv
  //  BenchGLConvolution<float16_t>(16, 16, 3, 3, 1280, 720, 0, 1, false);
  //  BenchGLConvolution<float16_t>(16, 16, 4, 4, 1280, 720, 0, 1, false);
  //  BenchGLConvolution<float16_t>(16, 16, 5, 5, 1280, 720, 0, 1, false);
  //  BenchGLConvolution<float16_t>(16, 16, 6, 6, 1280, 720, 0, 1, false);
  //  BenchGLConvolution<float16_t>(16, 16, 7, 7, 1280, 720, 0, 1, false);
  //  BenchGLConvolution<float16_t>(16, 16, 8, 8, 1280, 720, 0, 1, false);
  //  BenchGLConvolution<float16_t>(16, 16, 9, 9, 1280, 720, 0, 1, false);
  //
  //  BenchOp("Conv", 16, 16, 3, 3, 1, 1280, 720, false);
  //  BenchOp("Conv", 16, 16, 4, 4, 1, 1280, 720, false);
  //  BenchOp("Conv", 16, 16, 5, 5, 1, 1280, 720, false);
  //  BenchOp("Conv", 16, 16, 6, 6, 1, 1280, 720, false);
  //  BenchOp("Conv", 16, 16, 7, 7, 1, 1280, 720, false);
  //  BenchOp("Conv", 16, 16, 8, 8, 1, 1280, 720, false);
  //  BenchOp("Conv", 16, 16, 9, 9, 1, 1280, 720, false);

  //  BenchGLConvolution<float16_t>(16, 16, 3, 3, 80, 45, 0, 1, false);
  //  BenchGLConvolution<float16_t>(16, 16, 3, 3, 160, 90, 0, 1, false);
  //  BenchGLConvolution<float16_t>(16, 16, 3, 3, 320, 180, 0, 1, false);
  //  BenchGLConvolution<float16_t>(16, 16, 3, 3, 640, 360, 0, 1, false);
  //  BenchGLConvolution<float16_t>(16, 16, 3, 3, 1280, 720, 0, 1, false);
  //
  //  BenchOp("Conv", 16, 16, 3, 3, 1, 80, 45, false);
  //  BenchOp("Conv", 16, 16, 3, 3, 1, 160, 90, false);
  //  BenchOp("Conv", 16, 16, 3, 3, 1, 320, 180, false);
  //  BenchOp("Conv", 16, 16, 3, 3, 1, 640, 360, false);
  //  BenchOp("Conv", 16, 16, 3, 3, 1, 1280, 720, false);
  //
  //  BenchGLConvolution<float16_t>(128, 128, 3, 3, 14, 14, 0, 1, false);
  //  BenchGLConvolution<float16_t>(256, 256, 3, 3, 14, 14, 0, 1, false);
  //  BenchGLConvolution<float16_t>(128, 128, 3, 3, 28, 28, 0, 1, false);
  //  BenchGLConvolution<float16_t>(256, 256, 3, 3, 28, 28, 0, 1, false);
  //  BenchGLConvolution<float16_t>(128, 128, 3, 3, 56, 56, 0, 1, false);
  //  BenchGLConvolution<float16_t>(256, 256, 3, 3, 56, 56, 0, 1, false);
  //  BenchGLConvolution<float16_t>(64, 64, 7, 7, 128, 128, 0, 1, false);
  //
  //  BenchOp("Conv", 128, 128, 3, 3, 1, 14, 14, false);
  //  BenchOp("Conv", 256, 256, 3, 3, 1, 14, 14, false);
  //  BenchOp("Conv", 128, 128, 3, 3, 1, 28, 28, false);
  //  BenchOp("Conv", 256, 256, 3, 3, 1, 28, 28, false);
  //  BenchOp("Conv", 128, 128, 3, 3, 1, 56, 56, false);
  //  BenchOp("Conv", 256, 256, 3, 3, 1, 56, 56, false);
  //  BenchOp("Conv", 64, 64, 7, 7, 1, 128, 128, false);
}
