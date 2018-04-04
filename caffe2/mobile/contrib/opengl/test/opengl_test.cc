
#include "opengl_test.h"

#include "../core/GLContext.h"
#include "../core/GLImageAllocator.h"
#include "../core/GLLogging.h"
#include "../core/ImageAllocator.h"
#include "../core/arm_neon_support.h"
#include "../core/rewrite_net.h"
#include "../operators/gl_tiling_utils.h"

#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/core/workspace.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"

#ifdef CAFFE2_USE_MPSCNN
#include "caffe2/mobile/contrib/ios/mpscnn/mpscnn.h"
#endif

#define DEBUGGING false

namespace caffe2 {

template <class T>
float absolute_error(T t1, T t2) {
  return std::abs((float)t1 - (float)t2);
}

template <class T>
float relative_error(T t1, T t2) {
  return t2 != 0 ? absolute_error(t1, t2) / (float)t2 : 1;
}

// OpenGL: t1, CPU: t2
void checkError1D(const TensorCPU& t1, const TensorCPU& t2, float error) {
  CAFFE_ENFORCE_EQ(t1.size(), t2.size());
#if DEBUGGING
  gl_log(GL_LOG, "OpenGL output:\n");
  for (int i = 0; i < t1.size(); i++) {
    gl_log(GL_LOG, "%.5f\t", t1.template data<float>()[i]);
  }
  gl_log(GL_LOG, "\n");
  gl_log(GL_LOG, "CPU output:\n");
  for (int i = 0; i < t2.size(); i++) {
    gl_log(GL_LOG, "%.5f\t", t2.template data<float>()[i]);
  }
  gl_log(GL_LOG, "\n");

#else
  int count = 0;
  if (t1.template IsType<float>()) {
    for (auto i = 0; i < t1.size(); ++i) {
      const float t1_i = t1.template data<float>()[i];
      const float t2_i = t2.template data<float>()[i];

      if (!(absolute_error(t1_i, t2_i) <= error || relative_error(t1_i, t2_i) <= 0.08)) {
        gl_log(GL_ERR,
               "i: %d, GL: %.2f, CPU: %.2f, absolute error: %.2f, relative error: %.2f%%\n",
               i,
               t1_i,
               t2_i,
               absolute_error(t1_i, t2_i),
               relative_error(t1_i, t2_i) * 100);
        if (count++ == 10) {
          CAFFE_THROW("--- Test Failed ---");
        }
      }
    }
  }
#endif
}

// OpenGL: t1, CPU: t2
void checkError(const TensorCPU& t1, const TensorCPU& t2, float error) {
  CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
#if DEBUGGING
  gl_log(GL_LOG, "opengl_test output\n");
  gl_log(GL_LOG, "\nOpenGL output:\n");
  for (int i = 0; i < t1.size(); i++) {
    if (t1.ndim() > 2 && i % t1.dim(2) == 0) {
      gl_log(GL_LOG, "\n");
    }
    if (t1.ndim() > 2 && i != 0 && i % (4 * t2.dim(2) * t2.dim(3)) == 0) {
      gl_log(GL_LOG, "\n");
    }
    if (t1.template IsType<float>()) {
      const float t1_i = t1.template data<float>()[i];
      gl_log(GL_LOG, "%.3f\t", t1_i);
    } else if (t1.template IsType<uint8_t>()) {
      const uint8_t t1_i = t1.template data<uint8_t>()[i];
      gl_log(GL_LOG, "%.3d\t", (int)t1_i);
    }
  }

  gl_log(GL_LOG, "\nCPU output:\n");
  for (int i = 0; i < t2.size(); i++) {
    if (t2.ndim() > 2 && i % t2.dim(2) == 0)
      gl_log(GL_LOG, "\n");
    if (t2.ndim() > 2 && i != 0 && i % (4 * t2.dim(2) * t2.dim(3)) == 0)
      gl_log(GL_LOG, "\n");
    if (t2.template IsType<float>()) {
      const float t2_i = t2.template data<float>()[i];
      gl_log(GL_LOG, "%.3f\t", t2_i);
    } else if (t2.template IsType<uint8_t>()) {
      const uint8_t t2_i = t2.template data<uint8_t>()[i];
      gl_log(GL_LOG, "%.3d\t", (int)t2_i);
    }
  }
  gl_log(GL_LOG, "\n");
#else

  int count = 0;
  float maxError = 0, minError = 0;
  if (t1.template IsType<float>()) {
    for (auto i = 0; i < t1.size(); ++i) {
      const float t1_i = t1.template data<float>()[i];
      const float t2_i = t2.template data<float>()[i];
      if (!(absolute_error(t1_i, t2_i) <= error || relative_error(t1_i, t2_i) <= 0.08)) {
        if (count < 10) {
          gl_log(GL_ERR,
                 "i: %d, GL: %.2f, CPU: %.2f, absolute error: %.2f, relative error: %.2f%%\n",
                 i,
                 t1_i,
                 t2_i,
                 absolute_error(t1_i, t2_i),
                 relative_error(t1_i, t2_i) * 100);
        } else {
          CAFFE_THROW("--- Test Failed ---");
        }
        count++;
      }
      float err = t1_i - t2_i;
      if (err > maxError) {
        maxError = err;
      } else if (err < minError) {
        minError = err;
      }
    }
  } else if (t1.template IsType<uint8_t>()) {
    for (auto i = 0; i < t1.size(); ++i) {
      const uint8_t t1_i = t1.template data<uint8_t>()[i];
      const uint8_t t2_i = t2.template data<uint8_t>()[i];
      if (!(absolute_error(t1_i, t2_i) <= error || relative_error(t1_i, t2_i) <= 0.08)) {
        if (count < 10) {
          gl_log(GL_ERR,
                 "i: %d, GL: %d, CPU: %d, absolute error: %.2f, relative error: %.2f%%\n",
                 i,
                 t1_i,
                 t2_i,
                 absolute_error(t1_i, t2_i),
                 relative_error(t1_i, t2_i) * 100);
        } else {
          CAFFE_THROW("--- Test Failed ---");
        }
        count++;
      }
      float err = t1_i - t2_i;
      if (err > maxError) {
        maxError = err;
      } else if (err < minError) {
        minError = err;
      }
    }
  }
  gl_log(GL_LOG,
         "#errors = %d in %d, maxError = %f, minError = %f\n",
         count,
         (int)t1.size(),
         maxError,
         minError);
#endif
}

void testOpenGLCopyOps(int N, int C, int H, int W, float error, int tile_x = 1, int tile_y = 1) {
  LOG(INFO) << "OPENGLCopyFrom/To Test";
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);

    // Note: may overflow for half precision
    //    float *data = t->mutable_data<float>();
    //    for (int i = 0; i < t->size(); i++) {
    //      data[i] = i;
    //    }
  }

  NetDef netdef;
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
    op.set_type("CopyFromOpenGL");
    op.add_input("X_gl");
    op.add_output("Y_cpu");
  }

  ws.RunNetOnce(netdef);
  const auto& t1 = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // OpenGL
  const auto& t2 = ws.GetBlob("X_cpu")->Get<TensorCPU>(); // CPU
  CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());

  checkError(t1, t2, error);
}

typedef enum {
  AveragePool,
  MaxPool,
  Conv,
  ConvTranspose,
  ConvPRelu,
  ConvTransposePRelu,
  ConvRelu,
  ConvTransposeRelu
} PoolOp;

const char* glPoolOperationName[] = {"OpenGLAveragePool",
                                     "OpenGLMaxPool",
                                     "OpenGLConv",
                                     "OpenGLConvTranspose",
                                     "OpenGLConvPRelu",
                                     "OpenGLConvTransposePRelu",
                                     "OpenGLConvRelu",
                                     "OpenGLConvTransposeRelu"};

const char* cpuPoolOperationName[] = {"AveragePool",
                                      "MaxPool",
                                      "Conv",
                                      "ConvTranspose",
                                      "Conv",
                                      "ConvTranspose",
                                      "Conv",
                                      "ConvTranspose"};

void testOpenGLConv(int N,
                    int C,
                    int H,
                    int W,
                    int K, // output_channels
                    int kernel_h,
                    int kernel_w,
                    int pad,
                    int stride,
                    PoolOp poolOp,
                    float error,
                    bool random_input     = true,
                    int input_batch_size  = 1,
                    int output_batch_size = 1,
                    int input_tile_x      = 1,
                    int input_tile_y      = 1,
                    bool tiling           = false) {
  LOG(INFO) << "OpenGL Conv Test: "
            << "input C: " << C << ", output C: " << K << ", H: " << H << ", W: " << W
            << ", K: " << kernel_w << "x" << kernel_h << ", P: " << pad << ", S: " << stride
            << " Op: " << glPoolOperationName[poolOp];
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    if (random_input) {
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    } else {
      float* data = t->mutable_data<float>();
      for (int i = 0; i < t->size(); i++) {
        data[i] = 1;
      }
    }
#if 0
  gl_log(GL_LOG, "Input tensor:");
  for (int i = 0; i < t->size(); i++) {
    const float t1_i = t->data<float>()[i];
    if (i % t->dim(3) == 0)
      gl_log(GL_LOG, "\n");
    if (i % (4 * t->dim(2) * t->dim(3)) == 0)
      gl_log(GL_LOG, "-------------------------------\n");
    gl_log(GL_LOG, "%.3f\t", t1_i);
  }
  gl_log(GL_LOG, "\n\n");
#endif
  }

  if (poolOp != AveragePool && poolOp != MaxPool) {
    auto* t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
    if (poolOp != ConvTranspose && poolOp != ConvTransposePRelu && poolOp != ConvTransposeRelu) {
      t->Resize(K, C, kernel_h, kernel_w);
    } else {
      t->Resize(C, K, kernel_h, kernel_w);
    }
    CPUContext ctx;
    if (random_input) {
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    } else {
      float* data = t->mutable_data<float>();
      // Set the weights to all 1s
      //      for (int i = 0; i < t->size(); i++) {
      //        data[i] = 1;
      //      }

      // Set the weights to 1s, 2s, 3s... for channel 0, 1, 2, 3...
      int j = 0;
      for (int i = 0; i < t->size(); i++) {
        if (i % (C * kernel_h * kernel_w) == 0) {
          j++;
        }
        data[i] = j;
      }
    }

#if 0
    gl_log(GL_LOG, "Kernel (printing only the first line for each output channel):");
    for (int i = 0; i < t->size(); i++) {
      if (i == 0 || i % (t->dim(1) * t->dim(2) * t->dim(3)) == 0) {
        gl_log(GL_LOG, "\n");
        for (int j = 0; j < t->dim(3); j++) {
          const float t1_i = t->data<float>()[i + j];
          gl_log(GL_LOG, "%.3f\t", t1_i);
        }
      }
    }
    gl_log(GL_LOG, "\n");
#endif

    // bias
    {
      auto* t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
      t->Resize(K);
      CPUContext ctx;
      if (random_input) {
        math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
      } else {
        // Set bias to 1
        float* data = t->mutable_data<float>();
        for (int i = 0; i < t->size(); i++) {
          data[i] = i + 1;
        }
      }
#if 0
    gl_log(GL_LOG, "Bias:\n");
    for (int i = 0; i < t->size(); i++) {
      const float t1_i = t->data<float>()[i];
      gl_log(GL_LOG, "%.3f\t", t1_i);
    }
    gl_log(GL_LOG, "\n");
#endif
    }
  }

  if (poolOp == ConvPRelu || poolOp == ConvTransposePRelu) {
    auto* t = ws.CreateBlob("p")->GetMutable<TensorCPU>();
    t->Resize(K);
    CPUContext ctx;
    if (random_input) {
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    } else {
      // Set prelu scale to i + 1
      float* data = t->mutable_data<float>();
      for (int i = 0; i < t->size(); i++) {
        data[i] = -0.5;
      }
    }
  }

  NetDef netdef;
  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu");
    op.add_output("X_gl");
    {
      auto& arg = *(op.add_arg());
      arg.set_name("tile_x");
      arg.set_i(input_tile_x);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("tile_y");
      arg.set_i(input_tile_y);
    }
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type(glPoolOperationName[poolOp]);
    op.add_input("X_gl");
    if (poolOp != AveragePool && poolOp != MaxPool) {
      op.add_input("W");
      op.add_input("b");
    }
    if (poolOp == ConvPRelu || poolOp == ConvTransposePRelu) {
      op.add_input("p");
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NCHW");
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("kernel");
      arg.set_i(kernel_h);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("pad");
      arg.set_i(pad);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("stride");
      arg.set_i(stride);
    }
    if (poolOp != AveragePool && poolOp != MaxPool) {
      if (tiling) {
        {
          auto& arg = *(op.add_arg());
          arg.set_name("tiling");
          arg.set_i(1);
        }
      } else {
        {
          auto& arg = *(op.add_arg());
          arg.set_name("input_batch_size");
          arg.set_i(input_batch_size);
        }
        {
          auto& arg = *(op.add_arg());
          arg.set_name("output_batch_size");
          arg.set_i(output_batch_size);
        }
      }
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("is_last");
      arg.set_i(1);
    }
    op.add_output("Y_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Y_gl");
    op.add_output("Y_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type(cpuPoolOperationName[poolOp]);

    op.add_input("X_cpu");
    if (poolOp != AveragePool && poolOp != MaxPool) {
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
      arg.set_i(kernel_h);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("pad");
      arg.set_i(pad);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("stride");
      arg.set_i(stride);
    }
    op.add_output("Y_ref");
  }
  if (poolOp == ConvPRelu || poolOp == ConvTransposePRelu) {
    auto& op = *(netdef.add_op());
    op.set_type("PRelu");
    op.add_input("Y_ref");
    op.add_input("p");
    op.add_output("Y_ref");
    {
      auto& arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NCHW");
    }
  } else if (poolOp == ConvRelu || poolOp == ConvTransposeRelu) {
    auto& op = *(netdef.add_op());
    op.set_type("Relu");
    op.add_input("Y_ref");
    op.add_output("Y_ref");
    {
      auto& arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NCHW");
    }
  }

  ws.RunNetOnce(netdef);
  const auto& t1 = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // OpenGL
  const auto& t2 = ws.GetBlob("Y_ref")->Get<TensorCPU>(); // CPU

  checkError(t1, t2, error);
}

void testOpenGLPRelu(
    int N, int C, int H, int W, int prelu_size, int input_tile_x, int input_tile_y, float error) {
  LOG(INFO) << "OpenGL PRelu Test "
            << "C: " << C << ", H: " << H << ", W: " << W;
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    // Too noisy.
    math::RandGaussian<float, CPUContext>(t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }

  // prelu scale
  {
    auto* t = ws.CreateBlob("p")->GetMutable<TensorCPU>();
    t->Resize(prelu_size);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
  }

  NetDef netdef;
  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu");
    op.add_output("X_gl");
    {
      auto& arg = *(op.add_arg());
      arg.set_name("tile_x");
      arg.set_i(input_tile_x);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("tile_y");
      arg.set_i(input_tile_y);
    }
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("OpenGLPRelu");
    op.add_input("X_gl");
    op.add_input("p");
    op.add_output("Y_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Y_gl");
    op.add_output("Y_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("PRelu");
    op.add_input("X_cpu");
    op.add_input("p");
    auto& arg = *(op.add_arg());
    arg.set_name("order");
    arg.set_s("NCHW");
    op.add_output("Y_ref");
  }

  ws.RunNetOnce(netdef);
  const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // openGL
  const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>(); // CPU

  checkError(ws.GetBlob("Y_cpu")->Get<TensorCPU>(), ws.GetBlob("Y_ref")->Get<TensorCPU>(), error);
}

void testOpenGLRelu(int N, int C, int H, int W, int input_tile_x, int input_tile_y, float error) {
  LOG(INFO) << "OpenGL Relu Test "
            << "C: " << C << ", H: " << H << ", W: " << W;
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    // Too noisy.
    math::RandGaussian<float, CPUContext>(t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }

  NetDef netdef;
  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu");
    op.add_output("X_gl");
    {
      auto& arg = *(op.add_arg());
      arg.set_name("tile_x");
      arg.set_i(input_tile_x);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("tile_y");
      arg.set_i(input_tile_y);
    }
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("OpenGLRelu");
    op.add_input("X_gl");
    op.add_output("Y_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Y_gl");
    op.add_output("Y_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("Relu");
    op.add_input("X_cpu");
    auto& arg = *(op.add_arg());
    arg.set_name("order");
    arg.set_s("NCHW");
    op.add_output("Y_ref");
  }

  ws.RunNetOnce(netdef);
  const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // openGL
  const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>(); // CPU

  checkError(ws.GetBlob("Y_cpu")->Get<TensorCPU>(), ws.GetBlob("Y_ref")->Get<TensorCPU>(), error);
}

void testOpenGLAdd(int N, int C, int H, int W, float error = 0.1, int input_tile_x = 1, int input_tile_y = 1) {
  LOG(INFO) << "OpenGL Add Test "
            << "C: " << C << ", H: " << H << ", W: " << W;
  Workspace ws;
  {
    auto* t0 = ws.CreateBlob("X_cpu0")->GetMutable<TensorCPU>();
    t0->Resize(N, C, H, W);
    CPUContext ctx0;
    // Too noisy.
    math::RandGaussian<float, CPUContext>(t0->size(), 0, 30, t0->mutable_data<float>(), &ctx0);

    auto* t1 = ws.CreateBlob("X_cpu1")->GetMutable<TensorCPU>();
    t1->Resize(N, C, H, W);
    CPUContext ctx1;
    // Too noisy.
    math::RandGaussian<float, CPUContext>(t1->size(), 0, 30, t1->mutable_data<float>(), &ctx1);
  }

  NetDef netdef;
  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu0");
    op.add_output("X_gl0");
    {
      auto& arg = *(op.add_arg());
      arg.set_name("tile_x");
      arg.set_i(input_tile_x);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("tile_y");
      arg.set_i(input_tile_y);
    }
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu1");
    op.add_output("X_gl1");
    {
      auto& arg = *(op.add_arg());
      arg.set_name("tile_x");
      arg.set_i(input_tile_x);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("tile_y");
      arg.set_i(input_tile_y);
    }
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("OpenGLAdd");
    op.add_input("X_gl0");
    op.add_input("X_gl1");
    op.add_output("Y_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Y_gl");
    op.add_output("Y_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("Add");
    op.add_input("X_cpu0");
    op.add_input("X_cpu1");
    auto& arg = *(op.add_arg());
    arg.set_name("order");
    arg.set_s("NCHW");
    op.add_output("Y_ref");
  }
  ws.RunNetOnce(netdef);
  const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // openGL
  const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>(); // CPU

  checkError(t1, t2, error);
}

void testOpenGLSub(int N, int C, int H, int W, float error = 0.1) {
  LOG(INFO) << "OpenGL Sub Test "
            << "C: " << C << ", H: " << H << ", W: " << W;

  Workspace ws;
  {
    auto* t0 = ws.CreateBlob("X_cpu0")->GetMutable<TensorCPU>();
    t0->Resize(N, C, H, W);
    CPUContext ctx0;
    // Too noisy.
    math::RandGaussian<float, CPUContext>(t0->size(), 0, 30, t0->mutable_data<float>(), &ctx0);

    auto* t1 = ws.CreateBlob("X_cpu1")->GetMutable<TensorCPU>();
    t1->Resize(N, C, H, W);
    CPUContext ctx1;
    // Too noisy.
    math::RandGaussian<float, CPUContext>(t1->size(), 0, 30, t1->mutable_data<float>(), &ctx1);
  }

  NetDef netdef;
  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu0");
    op.add_output("X_gl0");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu1");
    op.add_output("X_gl1");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("OpenGLSub");
    op.add_input("X_gl0");
    op.add_input("X_gl1");
    op.add_output("Y_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Y_gl");
    op.add_output("Y_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("Sub");
    op.add_input("X_cpu0");
    op.add_input("X_cpu1");
    auto& arg = *(op.add_arg());
    arg.set_name("order");
    arg.set_s("NCHW");
    op.add_output("Y_ref");
  }
  ws.RunNetOnce(netdef);
  const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // openGL
  const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>(); // CPU
  checkError(t2, t1, error);
}

void testOpenGLConcat(int N, std::vector<int> Cs, int H, int W, bool tiling = false, float error = 0.1) {
  LOG(INFO) << "OpenGL Concat Test "
            << "H: " << H << ", W: " << W;
  Workspace ws;
  for (int i = 0; i < Cs.size(); i++) {
    auto* t = ws.CreateBlob("X_cpu" + caffe2::to_string(i))->GetMutable<TensorCPU>();
    t->Resize(N, Cs[i], H, W);
    CPUContext ctx0;
    // Too noisy.
    math::RandGaussian<float, CPUContext>(t->size(), 0, 30, t->mutable_data<float>(), &ctx0);
  }

  NetDef netdef;
  for (int i = 0; i < Cs.size(); i++) {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu" + caffe2::to_string(i));
    op.add_output("X_gl" + caffe2::to_string(i));
    if (tiling) {
      int tile_x = 1, tile_y = 1;
      computeOutputTiles(Cs[i], tile_x, tile_y);
      printf("Cs[i] = %d, tile_x = %d, tile_y = %d\n", Cs[i], tile_x, tile_y);
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
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("OpenGLConcat");
    for (int i = 0; i < Cs.size(); i++) {
      op.add_input("X_gl" + caffe2::to_string(i));
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NCHW");
    }
    op.add_output("Y_gl");
    op.add_output("Y_gl_mask");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Y_gl");
    op.add_output("Y_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("Concat");
    for (int i = 0; i < Cs.size(); i++) {
      op.add_input("X_cpu" + caffe2::to_string(i));
    }
    auto& arg = *(op.add_arg());
    arg.set_name("order");
    arg.set_s("NCHW");
    op.add_output("Y_ref");
    op.add_output("Y_ref_mask");
  }
  ws.RunNetOnce(netdef);
  const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // openGL
  const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>(); // CPU

  checkError(ws.GetBlob("Y_cpu")->Get<TensorCPU>(), ws.GetBlob("Y_ref")->Get<TensorCPU>(), error);
}

void testOpenGLSigmoid(int N, int C, int H, int W, float error) {
  LOG(INFO) << "OpenGL Sigmoid Test "
            << "C: " << C << ", H: " << H << ", W: " << W;
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    // Too noisy.
    math::RandGaussian<float, CPUContext>(t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }

  NetDef netdef;
  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu");
    op.add_output("X_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("OpenGLSigmoid");
    op.add_input("X_gl");
    op.add_output("Y_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Y_gl");
    op.add_output("Y_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("Sigmoid");
    op.add_input("X_cpu");
    auto& arg = *(op.add_arg());
    arg.set_name("order");
    arg.set_s("NCHW");
    op.add_output("Y_ref");
  }

  ws.RunNetOnce(netdef);
  const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // openGL
  const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>(); // CPU

  checkError(ws.GetBlob("Y_cpu")->Get<TensorCPU>(), ws.GetBlob("Y_ref")->Get<TensorCPU>(), error);
}

void testOpenGLTanh(int N, int C, int H, int W, float error) {
  LOG(INFO) << "OpenGL Tanh Test "
            << "C: " << C << ", H: " << H << ", W: " << W;
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(t->size(), 0, 2, t->mutable_data<float>(), &ctx);
  }

  NetDef netdef;
  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu");
    op.add_output("X_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("OpenGLTanh");
    op.add_input("X_gl");
    op.add_output("Y_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Y_gl");
    op.add_output("Y_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("Tanh");
    op.add_input("X_cpu");
    auto& arg = *(op.add_arg());
    arg.set_name("order");
    arg.set_s("NCHW");
    op.add_output("Y_ref");
  }

  ws.RunNetOnce(netdef);
  const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // openGL
  const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>(); // CPU

  checkError(ws.GetBlob("Y_cpu")->Get<TensorCPU>(), ws.GetBlob("Y_ref")->Get<TensorCPU>(), error);
}

void testOpenGLMul(int N, int C, int H, int W, float error) {
  LOG(INFO) << "OpenGL Mul Test "
            << "C: " << C << ", H: " << H << ", W: " << W;
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(t->size(), -10, 10, t->mutable_data<float>(), &ctx);
  }

  {
    auto* t = ws.CreateBlob("B")->GetMutable<TensorCPU>();
    t->Resize(1);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(t->size(), -10, 10, t->mutable_data<float>(), &ctx);
  }

  NetDef netdef;
  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu");
    op.add_output("X_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("OpenGLMul");
    op.add_input("X_gl");
    op.add_input("B");
    op.add_output("Y_gl");

    {
      auto& arg = *(op.add_arg());
      arg.set_name("broadcast");
      arg.set_i(1);
    }
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Y_gl");
    op.add_output("Y_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("Mul");
    op.add_input("X_cpu");
    op.add_input("B");

    {
      auto& arg = *(op.add_arg());
      arg.set_name("broadcast");
      arg.set_i(1);
    }

    op.add_output("Y_ref");
  }

  ws.RunNetOnce(netdef);
  const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // openGL
  const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>(); // CPU

  checkError(ws.GetBlob("Y_cpu")->Get<TensorCPU>(), ws.GetBlob("Y_ref")->Get<TensorCPU>(), error);
}

void testOpenGLSoftmax(int N, int D, float error, bool tiled = false) {
  LOG(INFO) << "OpenGL Softmax Test "
            << "N: " << N << " D: " << D << " Tiled:" << tiled;
  Workspace ws;
  auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
  {
    t->Resize(N, D);
    CPUContext ctx;
    // Too noisy.
    math::RandGaussian<float, CPUContext>(
        t->size(), 0, 1, t->mutable_data<float>(), &ctx);
  }

  NetDef netdef;
  {
    auto& op = *(netdef.add_op());
    op.set_type("Reshape");
    op.add_input("X_cpu");
    op.add_output("X_reshaped");
    op.add_output("old_shape");
    auto& arg = *(op.add_arg());
    arg.set_name("shape");
    if (tiled) {
      arg.add_ints(N);
      arg.add_ints(D);
      arg.add_ints(1);
      arg.add_ints(1);
    } else {
      arg.add_ints(N);
      arg.add_ints(1);
      arg.add_ints(D);
      arg.add_ints(1);
    }
  }
  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_reshaped");
    op.add_output("X_gl");
    if (tiled) {
      int tile_x = 1, tile_y = 1;
      squareFactors((D + 3) / 4, tile_x, tile_y);
      auto& argx = *(op.add_arg());
      argx.set_name("tile_x");
      argx.set_i(tile_x);
      auto& argy = *(op.add_arg());
      argy.set_name("tile_y");
      argy.set_i(tile_y);
    }
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("OpenGLSoftmax");
    op.add_input("X_gl");
    op.add_output("Y_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Y_gl");
    op.add_output("Y_cpu0");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("Reshape");
    op.add_input("Y_cpu0");
    op.add_output("Y_cpu");
    op.add_output("old_shape");
    auto& arg = *(op.add_arg());
    arg.set_name("shape");
    arg.add_ints(N);
    arg.add_ints(D);
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("Softmax");
    op.add_input("X_cpu");
    op.add_output("Y_ref");
  }

  ws.RunNetOnce(netdef);
  const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // OpenGL
  const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>(); // CPU
  checkError(ws.GetBlob("Y_cpu")->Get<TensorCPU>(), ws.GetBlob("Y_ref")->Get<TensorCPU>(), error);
}

void testOpenGLInstanceNorm(int N, int C, int H, int W, float error) {
  LOG(INFO) << "OpenGL InstanceNorm Test "
            << "C: " << C << ", H: " << H << ", W: " << W;
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    // Too noisy.
    math::RandGaussian<float, CPUContext>(t->size(), 0, 30, t->mutable_data<float>(), &ctx);
    //    for (auto i = 0; i < t->size(); ++i) {
    //      t->mutable_data<float>()[i] = 0.001;
    //    }
  }

  // scale
  {
    auto* t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
    t->Resize(C);
    CPUContext ctx;
    for (auto i = 0; i < t->size(); ++i) {
      t->mutable_data<float>()[i] = (i + 1) / t->size();
    }
  }
  // bias
  {
    auto* t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
    t->Resize(C);
    CPUContext ctx;
    for (auto i = 0; i < t->size(); ++i) {
      t->mutable_data<float>()[i] = 8 - 2 * i;
    }
  }

  NetDef netdef;
  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu");
    op.add_output("X_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("OpenGLInstanceNorm");
    op.add_input("X_gl");
    op.add_input("W");
    op.add_input("b");
    op.add_output("Y_gl");
    op.add_output("Mean_gl");
    op.add_output("InvStdev_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Y_gl");
    op.add_output("Y_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Mean_gl");
    op.add_output("Mean_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("InvStdev_gl");
    op.add_output("InvStdev_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("InstanceNorm");
    op.add_input("X_cpu");
    op.add_input("W");
    op.add_input("b");
    auto& arg = *(op.add_arg());
    arg.set_name("order");
    arg.set_s("NCHW");
    op.add_output("Y_ref");
    op.add_output("Mean_ref");
    op.add_output("InvStdev_ref");
  }

  ws.RunNetOnce(netdef);
  const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // openGL
  const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>(); // CPU

  LOG(INFO) << "Check mean";
  checkError1D(
      ws.GetBlob("Mean_cpu")->Get<TensorCPU>(), ws.GetBlob("Mean_ref")->Get<TensorCPU>(), 0.001);
  LOG(INFO) << "Check inv_stdev";
  checkError1D(ws.GetBlob("InvStdev_cpu")->Get<TensorCPU>(),
               ws.GetBlob("InvStdev_ref")->Get<TensorCPU>(),
               0.001);
  LOG(INFO) << "Check instance norm";
  checkError(ws.GetBlob("Y_cpu")->Get<TensorCPU>(), ws.GetBlob("Y_ref")->Get<TensorCPU>(), error);
}

void testOpenGLInstanceNormPRelu(int N, int C, int H, int W, float error) {
  LOG(INFO) << "OpenGL InstanceNormPRelu Test "
            << "C: " << C << ", H: " << H << ", W: " << W;
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    // Too noisy.
    math::RandGaussian<float, CPUContext>(t->size(), 0, 30, t->mutable_data<float>(), &ctx);
    //    for (auto i = 0; i < t->size(); ++i) {
    //      t->mutable_data<float>()[i] = 0.001;
    //    }
  }

  // scale
  {
    auto* t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
    t->Resize(C);
    CPUContext ctx;
    for (auto i = 0; i < t->size(); ++i) {
      t->mutable_data<float>()[i] = (i + 1) / t->size();
    }
  }
  // bias
  {
    auto* t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
    t->Resize(C);
    CPUContext ctx;
    for (auto i = 0; i < t->size(); ++i) {
      t->mutable_data<float>()[i] = 8 - 2 * i;
    }
  }
  // prelu scale
  {
    auto* t = ws.CreateBlob("p")->GetMutable<TensorCPU>();
    t->Resize(C);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
  }

  NetDef netdef;
  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu");
    op.add_output("X_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("OpenGLInstanceNormPRelu");
    op.add_input("X_gl");
    op.add_input("W");
    op.add_input("b");
    op.add_input("p");
    op.add_output("Y_gl");
    op.add_output("Mean_gl");
    op.add_output("InvStdev_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Y_gl");
    op.add_output("Y_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Mean_gl");
    op.add_output("Mean_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("InvStdev_gl");
    op.add_output("InvStdev_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("InstanceNorm");
    op.add_input("X_cpu");
    op.add_input("W");
    op.add_input("b");
    auto& arg = *(op.add_arg());
    arg.set_name("order");
    arg.set_s("NCHW");
    op.add_output("Y_ref");
    op.add_output("Mean_ref");
    op.add_output("InvStdev_ref");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("PRelu");
    op.add_input("Y_ref");
    op.add_input("p");
    auto& arg = *(op.add_arg());
    arg.set_name("order");
    arg.set_s("NCHW");
    op.add_output("Y_ref");
  }

  ws.RunNetOnce(netdef);
  const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // openGL
  const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>(); // CPU

  LOG(INFO) << "Check mean";
  checkError1D(
      ws.GetBlob("Mean_cpu")->Get<TensorCPU>(), ws.GetBlob("Mean_ref")->Get<TensorCPU>(), 0.001);
  LOG(INFO) << "Check inv_stdev";
  checkError1D(ws.GetBlob("InvStdev_cpu")->Get<TensorCPU>(),
               ws.GetBlob("InvStdev_ref")->Get<TensorCPU>(),
               0.001);
  LOG(INFO) << "Check instance norm";
  checkError(ws.GetBlob("Y_cpu")->Get<TensorCPU>(), ws.GetBlob("Y_ref")->Get<TensorCPU>(), error);
}

void OpenGL_speedtest(int N,
                      int C,
                      int H,
                      int W,
                      int K,
                      int kernel_h,
                      int kernel_w,
                      int pad,
                      float error,
                      bool random_input = true) {
  LOG(INFO) << "OpenGL Conv Speed Test "
            << " C: " << C << " H: " << H << " W: " << W;
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    if (random_input) {
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    } else {
      float* data = t->mutable_data<float>();
      for (int i = 0; i < t->size(); i++) {
        data[i] = 1;
      }
    }
  }

  {
    auto* t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
    t->Resize(K, C, kernel_h, kernel_w);
    CPUContext ctx;
    if (random_input) {
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    } else {
      float* data = t->mutable_data<float>();
      for (int i = 0; i < t->size(); i++) {
        data[i] = 1;
      }
    }
  }

  {
    auto* t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
    t->Resize(K);
    CPUContext ctx;
    if (random_input) {
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    } else {
      float* data = t->mutable_data<float>();
      for (int i = 0; i < t->size(); i++) {
        data[i] = 1;
      }
    }
  }

  NetDef netdef;
  netdef.set_name("Test net");
  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu");
    op.add_output("X_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("OpenGLConv");
    op.add_input("X_gl");
    op.add_input("W");
    op.add_input("b");
    {
      auto& arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NCHW");
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("kernel");
      arg.set_i(kernel_h);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("pad");
      arg.set_i(pad);
    }
    op.add_output("Y_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Y_gl");
    op.add_output("Y_cpu");
  }

  CAFFE_ENFORCE(ws.RunNetOnce(netdef));
  caffe2::NetBase* net = ws.CreateNet(netdef);
  CHECK_NOTNULL(net);
  CAFFE_ENFORCE(net->Run());
  net->TEST_Benchmark(1, 4, true);
}

void testOpenGLPadImage(
    int N, int C, int H, int W, int pad_l, int pad_r, int pad_t, int pad_b, float error) {
  LOG(INFO) << "OpenGLPadImage Test";
  {
    Workspace ws;
    {
      auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
      t->Resize(N, C, H, W);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
      //      for (auto i = 0; i < t->size(); ++i) {
      //        t->mutable_data<float>()[i] = i + 1;
      //      }
    }

    NetDef netdef;
    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyToOpenGL");
      op.add_input("X_cpu");
      op.add_output("X_gl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("OpenGLPadImage");
      op.add_input("X_gl");
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_l");
        arg.set_i(pad_l);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_r");
        arg.set_i(pad_r);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_t");
        arg.set_i(pad_t);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_b");
        arg.set_i(pad_b);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("mode");
        arg.set_s("reflect");
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("is_last");
        arg.set_i(1);
      }
      op.add_output("Y_gl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyFromOpenGL");
      op.add_input("Y_gl");
      op.add_output("Y_cpu");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("PadImage");
      op.add_input("X_cpu");
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_l");
        arg.set_i(pad_l);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_r");
        arg.set_i(pad_r);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_t");
        arg.set_i(pad_t);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_b");
        arg.set_i(pad_b);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("mode");
        arg.set_s("reflect");
      }
      op.add_output("Y_ref");
    }

    ws.RunNetOnce(netdef);

    const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // opengl
    const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>(); // cpu
    checkError(t2, t1, error);
  }
}

void testOpenGLResize(int N,
                      int C,
                      int H,
                      int W,
                      int width_scale,
                      int height_scale,
                      float error,
                      int input_tile_x = 1,
                      int input_tile_y = 1) {
  LOG(INFO) << "OpenGLResize Test";
  {
    Workspace ws;
    {
      auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
      t->Resize(N, C, H, W);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }

    NetDef netdef;
    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyToOpenGL");
      op.add_input("X_cpu");
      op.add_output("X_gl");
      {
        auto& arg = *(op.add_arg());
        arg.set_name("tile_x");
        arg.set_i(input_tile_x);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("tile_y");
        arg.set_i(input_tile_y);
      }
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("OpenGLResizeNearest");
      op.add_input("X_gl");
      {
        auto& arg = *(op.add_arg());
        arg.set_name("width_scale");
        arg.set_f(width_scale);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("height_scale");
        arg.set_f(height_scale);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("is_last");
        arg.set_i(1);
      }
      op.add_output("Y_gl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyFromOpenGL");
      op.add_input("Y_gl");
      op.add_output("Y_cpu");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("ResizeNearest");
      op.add_input("X_cpu");
      {
        auto& arg = *(op.add_arg());
        arg.set_name("width_scale");
        arg.set_f(width_scale);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("height_scale");
        arg.set_f(height_scale);
      }
      op.add_output("Y_ref");
    }

    ws.RunNetOnce(netdef);

    const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // opengl
    const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>(); // cpu
    checkError(t2, t1, error);
  }
}

void testOpenGLPreprocess(int N, int C, int H, int W, float error) {
  LOG(INFO) << "OpenGL Preprocess Test";
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, H, W, C);
    CPUContext ctx;
    for (auto i = 0; i < t->size(); ++i) {
      t->mutable_data<uint8_t>()[i] = rand() % 255;
    }
  }

  {
    auto* t = ws.CreateBlob("mean")->GetMutable<TensorCPU>();
    t->Resize(3);
    CPUContext ctx;
    t->mutable_data<float>()[0] = 100;
    t->mutable_data<float>()[1] = 50;
    t->mutable_data<float>()[2] = 150;
  }

  NetDef netdef;

  {
    auto& op = *(netdef.add_op());
    op.set_type("OpenGLTensorToTextureStylizerPreprocess");
    op.add_input("X_cpu");
    op.add_input("mean");
    {
      auto& arg = *(op.add_arg());
      arg.set_name("noise_std");
      arg.set_f(0.00001);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("noise_size");
      arg.set_i(512);
    }

    op.add_output("Y_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Y_gl");
    op.add_output("Y_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("PackedInt8BGRANHWCToNCHWCStylizerPreprocess");
    op.add_input("X_cpu");
    op.add_input("mean");
    {
      auto& arg = *(op.add_arg());
      arg.set_name("noise_std");
      arg.set_f(0.00001);
    }
    {
      auto& arg = *(op.add_arg());
      arg.set_name("noise_size");
      arg.set_i(512);
    }
    op.add_output("Y_ref");
  }

  ws.RunNetOnce(netdef);
  const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // openGL
  const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>(); // cpu
  checkError(t2, t1, error);
}

void testOpenGLDeprocess(int N, int C, int H, int W, float error) {
  LOG(INFO) << "OpenGLDeprocess Test";
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    for (auto i = 0; i < t->size(); ++i) {
      t->mutable_data<float>()[i] = rand() % 1000 - 500;
    }
  }

  {
    auto* t = ws.CreateBlob("mean")->GetMutable<TensorCPU>();
    t->Resize(3);
    CPUContext ctx;
    t->mutable_data<float>()[0] = 30;
    t->mutable_data<float>()[1] = 40;
    t->mutable_data<float>()[2] = 50;
  }

  NetDef netdef;

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu");
    op.add_output("X_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("OpenGLTextureToTensorStylizerDeprocess");
    op.add_input("X_gl");
    op.add_input("mean");
    op.add_output("Y_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("BRGNCHWCToPackedInt8BGRAStylizerDeprocess");
    op.add_input("X_cpu");
    op.add_input("mean");
    op.add_output("Y_ref");
  }

  ws.RunNetOnce(netdef);
  const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
  const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();
  checkError(t2, t1, error);
}

void testOpenGLNormPlanarYUV(int N, int C, int H, int W, float error) {
  LOG(INFO) << "OpenGLNormPlanarYUV Test";
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, 3, H, W);
    CPUContext ctx;
    for (auto i = 0; i < t->size(); ++i) {
      t->mutable_data<float>()[i] = rand() % 1000 - 500;
    }
  }

  {
    auto* t = ws.CreateBlob("mean")->GetMutable<TensorCPU>();
    t->Resize(1, 3);
    CPUContext ctx;
    t->mutable_data<float>()[0] = 30;
    t->mutable_data<float>()[1] = 40;
    t->mutable_data<float>()[2] = 50;
  }

  {
    auto* t = ws.CreateBlob("stdev")->GetMutable<TensorCPU>();
    t->Resize(1, 3);
    CPUContext ctx;
    t->mutable_data<float>()[0] = 6;
    t->mutable_data<float>()[1] = 7;
    t->mutable_data<float>()[2] = 8;
  }

  NetDef netdef;

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu");
    op.add_output("X_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("OpenGLNormalizePlanarYUV");
    op.add_input("X_gl");
    op.add_input("mean");
    op.add_input("stdev");
    op.add_output("Y_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("Y_gl");
    op.add_output("Y_cpu");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("NormalizePlanarYUV");
    op.add_input("X_cpu");
    op.add_input("mean");
    op.add_input("stdev");
    op.add_output("Y_ref");
  }

  ws.RunNetOnce(netdef);
  const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
  const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();
  checkError(t2, t1, error);
}

void OpenGL_copyops_speedtest(int N,
                              int C,
                              int H,
                              int W,
                              int K,
                              int kernel_h,
                              int kernel_w,
                              int pad,
                              float error,
                              bool random_input = true) {
  LOG(INFO) << "OpenGL CopyOps Speed Test";
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    if (random_input) {
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    } else {
      float* data = t->mutable_data<float>();
      for (int i = 0; i < t->size(); i++) {
        data[i] = 1;
      }
    }
  }

  {
    auto* t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
    t->Resize(K, C, kernel_h, kernel_w);
    CPUContext ctx;
    if (random_input) {
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    } else {
      float* data = t->mutable_data<float>();
      for (int i = 0; i < t->size(); i++) {
        data[i] = 1;
      }
    }
  }

  {
    auto* t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
    t->Resize(K);
    CPUContext ctx;
    if (random_input) {
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    } else {
      float* data = t->mutable_data<float>();
      for (int i = 0; i < t->size(); i++) {
        data[i] = 1;
      }
    }
  }

  NetDef netdef;
  netdef.set_name("Test net");
  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyToOpenGL");
    op.add_input("X_cpu");
    op.add_output("X_gl");
  }

  {
    auto& op = *(netdef.add_op());
    op.set_type("CopyFromOpenGL");
    op.add_input("X_gl");
    op.add_output("Y_cpu");
  }

  caffe2::NetBase* net = ws.CreateNet(netdef);
  CHECK_NOTNULL(net);
  net->TEST_Benchmark(1, 4, true);
}

static NetDef truncateAfter(NetDef def, size_t idx) {
  // idx = 0, net = 10 -> remove 9
  // idx = 0, net = 1 -> remove 0
  const auto toRemove = def.op_size() - idx - 1;
  for (auto i = 0; i < toRemove; ++i) {
    def.mutable_op()->RemoveLast();
  }
  CHECK_EQ(def.op_size(), idx + 1);
  return def;
}

void compareModelsForOpenGL(std::string name,
                            const NetDef& initNet,
                            NetDef predictNet,
                            int width,
                            int height,
                            int channel,
                            std::string input_type,
                            std::string input_order) {

  if (name == "styleTransfer") {
    for (int i = 0; i < predictNet.mutable_op(0)->arg_size(); i++) {
      auto* arg = predictNet.mutable_op(0)->mutable_arg(i);
      if (arg->name() == "noise_std") {
        arg->set_f(0);
      }
    }
  }

  for (auto i = 0; i < predictNet.op_size(); ++i) {
    auto truncatedPredictNet = truncateAfter(predictNet, i);

    // Change the last blob to external_output(0) for the predict net
    auto output_blob = "_OUTPUT_BLOB__";
    truncatedPredictNet.set_external_output(0, output_blob);
    truncatedPredictNet.mutable_op(truncatedPredictNet.op_size() - 1)->set_output(0, output_blob);

    NetDef truncatedOpenGLPredictNet = rewritePredictNetForOpenGL(truncatedPredictNet);

    //    LOG(INFO) << "truncatedPredictNet";
    //    dumpDefForOpenGL(truncatedPredictNet);
    //
    LOG(INFO) << "truncatedOpenGLPredictNet";
    dumpDefForOpenGL(truncatedOpenGLPredictNet);

    CPUContext ctx;
    Workspace cws;
    cws.RunNetOnce(initNet);

    auto* t_cpu = cws.CreateBlob(truncatedPredictNet.external_input(0))->GetMutable<TensorCPU>();
    if (name == "styleTransfer") {
      CAFFE_ENFORCE_EQ(input_order, "NHWC");
      CAFFE_ENFORCE_EQ(input_type, "uint8_t");
      t_cpu->Resize(1, height, width, channel);
      for (auto i = 0; i < t_cpu->size(); ++i) {
        t_cpu->mutable_data<uint8_t>()[i] = i % 255;
      }
    } else if (name == "segmentation") {
      CAFFE_ENFORCE_EQ(input_order, "NCHW");
      CAFFE_ENFORCE_EQ(input_type, "float");
      t_cpu->Resize(1, channel, height, width);
      float* input = t_cpu->mutable_data<float>();
      const int size = width * height;
      // Limit input range to YUV
      math::RandGaussian<float, CPUContext>(size, 0.5, 0.15, input, &ctx); // Y: 0 ~ 1
      math::RandGaussian<float, CPUContext>(size, 0, 0.12, input + size, &ctx); // U: -0.436 ~ 0.436
      math::RandGaussian<float, CPUContext>(
          size, 0, 0.2, input + 2 * size, &ctx); // V: -0.615 ~ 0.615
    } else if (name == "denoiser") {
      CAFFE_ENFORCE_EQ(input_order, "NCHW");
      CAFFE_ENFORCE_EQ(input_type, "float");
      t_cpu->Resize(1, channel, height, width);
      float* input = t_cpu->mutable_data<float>();
      const int spatial_size = width * height;
      math::RandGaussian<float, CPUContext>(spatial_size, 0, 0.33, input, &ctx); // R Channel
      math::RandGaussian<float, CPUContext>(
          spatial_size, 0, 0.33, input + spatial_size, &ctx); // G Channel
      math::RandGaussian<float, CPUContext>(
          spatial_size, 0, 0.33, input + 2 * spatial_size, &ctx); // B Channel
      // Clamp Range of input [-1, +1]
      for (auto i = 0; i < t_cpu->size(); ++i) {
        input[i] = input[i] > 1 ? 1 : input[i] < -1 ? -1 : input[i];
      }
    } else {
      CAFFE_THROW("CompareModels only works with style transfer and segmentation now");
    }

    Workspace mws;
    mws.RunNetOnce(initNet);

    auto* t_gl =
        mws.CreateBlob(truncatedOpenGLPredictNet.external_input(0))->GetMutable<TensorCPU>();
    if (name == "styleTransfer") {
      CAFFE_ENFORCE_EQ(input_order, "NHWC");
      CAFFE_ENFORCE_EQ(input_type, "uint8_t");
      t_gl->Resize(1, height, width, channel);
      uint8_t* input = t_gl->mutable_data<uint8_t>();
      memcpy(input, t_cpu->mutable_data<uint8_t>(), t_cpu->capacity_nbytes());
    } else if (name == "segmentation") {
      CAFFE_ENFORCE_EQ(input_order, "NCHW");
      CAFFE_ENFORCE_EQ(input_type, "float");
      t_gl->Resize(1, channel, height, width);
      float* input = t_gl->mutable_data<float>();
      memcpy(input, t_cpu->mutable_data<float>(), t_cpu->capacity_nbytes());
    } else if (name == "denoiser") {
      CAFFE_ENFORCE_EQ(input_order, "NCHW");
      CAFFE_ENFORCE_EQ(input_type, "float");
      t_gl->Resize(1, channel, height, width);
      float* input = t_gl->mutable_data<float>();
      memcpy(input, t_cpu->mutable_data<float>(), t_cpu->capacity_nbytes());
    }

    cws.RunNetOnce(truncatedPredictNet);
    mws.RunNetOnce(truncatedOpenGLPredictNet);

    const auto m_name =
        truncatedOpenGLPredictNet.op(truncatedOpenGLPredictNet.op_size() - 1).output(0);
    const auto c_name = truncatedPredictNet.op(truncatedPredictNet.op_size() - 1).output(0);

    LOG(INFO) << "Checking correspondence for name: " << m_name << ", idx: " << i;
    {
      const auto& mt = mws.GetBlob(m_name)->Get<TensorCPU>(); // GPU
      const auto& ct = cws.GetBlob(c_name)->Get<TensorCPU>(); // CPU
      if (name == "denoiser") {
        checkError(mt, ct, 0.02); // 1% of Scale
        LOG(INFO) << "Error Check Completed for Denoiser Layer: " << i;
      } else {
        checkError(mt, ct, 1);
      }
    }
  }
}

void compareBatchedToTiledModels(std::string name,
                                 const NetDef& initNet,
                                 NetDef predictNet,
                                 int width,
                                 int height,
                                 int channel,
                                 std::string input_type,
                                 std::string input_order) {

  if (name == "styleTransfer") {
    for (int i = 0; i < predictNet.mutable_op(0)->arg_size(); i++) {
      auto* arg = predictNet.mutable_op(0)->mutable_arg(i);
      if (arg->name() == "noise_std") {
        arg->set_f(0);
      }
    }
  }

  for (auto i = 19; i < predictNet.op_size(); ++i) {
    auto truncatedPredictNet = truncateAfter(predictNet, i);

    // Change the last blob to external_output(0) for the predict net
    auto output_blob = "_OUTPUT_BLOB__";
    truncatedPredictNet.set_external_output(0, output_blob);
    truncatedPredictNet.mutable_op(truncatedPredictNet.op_size() - 1)->set_output(0, output_blob);

    NetDef bachedNet = rewritePredictNetForOpenGL(truncatedPredictNet, false, false);
    NetDef tiledNet = rewritePredictNetForOpenGL(truncatedPredictNet, false, true);

    LOG(INFO) << "truncatedPredictNet";
    dumpDefForOpenGL(truncatedPredictNet);

    LOG(INFO) << "truncatedOpenGLPredictNet";
    dumpDefForOpenGL(bachedNet);

    CPUContext ctx;

    Workspace tws;
    tws.RunNetOnce(initNet);

    auto* t_batch = tws.CreateBlob(bachedNet.external_input(0))->GetMutable<TensorCPU>();
    if (name == "styleTransfer") {
      CAFFE_ENFORCE_EQ(input_order, "NHWC");
      CAFFE_ENFORCE_EQ(input_type, "uint8_t");
      t_batch->Resize(1, height, width, channel);
      for (auto i = 0; i < t_batch->size(); ++i) {
        t_batch->mutable_data<uint8_t>()[i] = i % 255;
      }
    } else if (name == "segmentation") {
      CAFFE_ENFORCE_EQ(input_order, "NCHW");
      CAFFE_ENFORCE_EQ(input_type, "float");
      t_batch->Resize(1, channel, height, width);
      float* input = t_batch->mutable_data<float>();
      const int size = width * height;
      // Limit input range to YUV
      math::RandGaussian<float, CPUContext>(size, 0.5, 0.15, input, &ctx); // Y: 0 ~ 1
      math::RandGaussian<float, CPUContext>(size, 0, 0.12, input + size, &ctx); // U: -0.436 ~ 0.436
      math::RandGaussian<float, CPUContext>(
          size, 0, 0.2, input + 2 * size, &ctx); // V: -0.615 ~ 0.615
    } else {
      CAFFE_THROW("CompareModels only works with style transfer and segmentation now");
    }

    Workspace bws;
    bws.RunNetOnce(initNet);

    auto* t_tiling = bws.CreateBlob(tiledNet.external_input(0))->GetMutable<TensorCPU>();
    if (name == "styleTransfer") {
      CAFFE_ENFORCE_EQ(input_order, "NHWC");
      CAFFE_ENFORCE_EQ(input_type, "uint8_t");
      t_tiling->Resize(1, height, width, channel);
      uint8_t* input = t_tiling->mutable_data<uint8_t>();
      memcpy(input, t_batch->mutable_data<uint8_t>(), t_batch->capacity_nbytes());

    } else if (name == "segmentation") {
      CAFFE_ENFORCE_EQ(input_order, "NCHW");
      CAFFE_ENFORCE_EQ(input_type, "float");
      t_tiling->Resize(1, channel, height, width);
      float* input = t_tiling->mutable_data<float>();
      memcpy(input, t_batch->mutable_data<float>(), t_batch->capacity_nbytes());
    }

    bws.RunNetOnce(bachedNet);
    tws.RunNetOnce(tiledNet);

    const auto batch_name = bachedNet.op(bachedNet.op_size() - 1).output(0);
    const auto tile_name = tiledNet.op(tiledNet.op_size() - 1).output(0);

    LOG(INFO) << "Checking correspondence for name: " << batch_name << ", idx: " << i;
    {
      const auto& bt = bws.GetBlob(batch_name)->Get<TensorCPU>(); // GPU
      const auto& tt = tws.GetBlob(tile_name)->Get<TensorCPU>(); // CPU
      checkError(bt, tt, 0.01);
    }
  }
}

int runModelBenchmarks(caffe2::NetDef& init_net,
                       caffe2::NetDef& predict_net,
                       int warm_up_runs,
                       int main_runs,
                       int channel,
                       int height,
                       int width,
                       std::string input_type,
                       std::string input_order,
                       std::string engine, // "CPU", "OPENGL", or "MPSCNN"
                       bool run_individual,
                       bool use_texture_input,
                       bool use_tiling,
                       bool run_fusion) {
  std::unique_ptr<caffe2::Workspace> workspace(new caffe2::Workspace());

  // caffe2::dumpDefForOpenGL(init_net);
  caffe2::dumpDefForOpenGL(predict_net);

  CAFFE_ENFORCE(workspace->RunNetOnce(init_net));
  caffe2::NetDef net_def;

  // rewrite network
  if (engine == "CPU") {
    net_def.CopyFrom(predict_net);
  } else if (engine == "OPENGL") {
    if (!caffe2::tryConvertToOpenGL(init_net, predict_net, &net_def, use_texture_input, use_tiling, run_fusion)) {
      CAFFE_THROW("Failed to convert to openGL. Benchmark failed to run");
      return -1;
    }
  } else if (engine == "MPSCNN") {
#ifdef CAFFE2_USE_MPSCNN
    if (!caffe2::tryConvertToMPSCNN(init_net, predict_net, &net_def)) {
      CAFFE_THROW("Failed to convert to MPSCNN. Benchmark failed to run");
      return -1;
    }
#else
    CAFFE_THROW("MPSCNN not enabled. Benchmark failed to run");
    return -1;
#endif
  } else {
    CAFFE_THROW("Unsupported engine. Benchmark failed to run");
    return -1;
  }

  if (!net_def.has_name()) {
    net_def.set_name("benchmark");
  }
  caffe2::NetBase* net = workspace->CreateNet(net_def);

  // create input blob
  if (engine == "CPU" || engine == "MPSCNN" || !use_texture_input) {
    caffe2::TensorCPU* b;
    if (!net_def.external_input_size()) {
      b = workspace->CreateBlob("data")->GetMutable<caffe2::TensorCPU>();
    } else {
      b = workspace->CreateBlob(net_def.external_input(0))->GetMutable<caffe2::TensorCPU>();
    }

    if (input_order == "NCHW") {
      b->Resize(std::vector<int32_t>(
          {1, static_cast<int>(channel), static_cast<int>(height), static_cast<int>(width)}));
    } else if (input_order == "NHWC") {
      b->Resize(std::vector<int32_t>(
          {1, static_cast<int>(height), static_cast<int>(width), static_cast<int>(channel)}));
    } else {
      CAFFE_THROW("Unknown input order: ", input_order);
    }
    if (input_type == "uint8_t") {
      b->mutable_data<uint8_t>();
    } else if (input_type == "float") {
      b->mutable_data<float>();
    } else {
      CAFFE_THROW("Unknown input type: ", input_type);
    }
  } else {
    const int tile_x = 1, tile_y = 1;
    Blob* blob = nullptr;
    if (!net_def.external_input_size()) {
      blob = workspace->CreateBlob("data");
    } else {
      blob = workspace->CreateBlob(net_def.external_input(0));
    }
    if (input_type == "float") {
      ImageAllocator<float16_t> allocator;
      GLImageVector<float16_t>* output_image = allocator.newImage(1,
                                                                  width,
                                                                  height,
                                                                  channel,
                                                                  tile_x,
                                                                  tile_y,
#if CAFFE2_IOS
                                                                  true
#else
                                                                  false
#endif
      );
      blob->Reset(output_image);
      for (auto& texture : (*output_image)[0]->textures) {
        texture->map_load([&](void* buffer,
                              size_t width,
                              size_t height,
                              size_t stride,
                              size_t channels,
                              const GLTexture::Type& type) {});
      }
    } else {
      ImageAllocator<uint8_t> allocator;
      GLImageVector<uint8_t>* output_image = allocator.newImage(1,
                                                                width,
                                                                height,
                                                                channel,
                                                                tile_x,
                                                                tile_y,
#if CAFFE2_IOS
                                                                true
#else
                                                                false
#endif
      );
      blob->Reset(output_image);
      for (auto& texture : (*output_image)[0]->textures) {
        texture->map_load([&](void* buffer,
                              size_t width,
                              size_t height,
                              size_t stride,
                              size_t channels,
                              const GLTexture::Type& type) {});
      }
    }
  }

  // run benchmark
  if (engine == "CPU" || engine == "MPSCNN") {
    CHECK_NOTNULL(net);
    CAFFE_ENFORCE(net->Run());
    net->TEST_Benchmark(warm_up_runs, main_runs, run_individual);
  } else if (engine == "OPENGL") {
    CHECK_NOTNULL(net);
    CAFFE_ENFORCE(net->Run());

    for (int i = 0; i < warm_up_runs; i++) {
      net->Run();
    }
    glFinish();

    Timer timer;
    timer.Start();
    for (int i = 0; i < main_runs; i++) {
      net->Run();
    }
    if (use_texture_input) {
      glFinish();
    }

    double iter_time = (double)timer.MilliSeconds() / main_runs;
    LOG(INFO) << "Main run finished. Milliseconds per iter: " << iter_time
              << ". Iters per second: " << 1000.0 / iter_time;

    if (run_individual) {
      std::vector<std::unique_ptr<caffe2::OperatorBase>> ops;

      for (auto& op : net_def.op()) {
        ops.push_back(CreateOperator(op, workspace.get()));
        ops.back()->Run(); // warm up
      }

      for (int k = 0; k < ops.size(); k++) {
        timer.Start();
        for (int i = 0; i < main_runs; i++) {
          ops[k]->Run();
        }
        glFinish();

        LOG(INFO) << "Operator #" << k << " " << net_def.op(k).type() << ": "
                  << (double)timer.MilliSeconds() / main_runs;
      }
    }
  }

  return 0;
}

template <typename T>
void testGLTextureTypes() {
  gl_log(GL_LOG, "Executing %s...\n", __PRETTY_FUNCTION__);

  GLImageAllocator<T>* allocator = GLImageAllocator<T>::newGLImageAllocator();

  GLImageVector<T>* image = allocator->newImage(1, 10, 10, 4, 1, 1, true);

  const GLTexture* texture = (*image)[0]->textures[0];

  texture->map_load([&](void* buffer,
                        size_t width,
                        size_t height,
                        size_t stride,
                        size_t channels,
                        const GLTexture::Type& type) {
    T* buffer_data = (T*)buffer;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        for (int c = 0; c < channels; c++) {
          buffer_data[channels * (y * stride + x) + c] = x + y;
        }
      }
    }
  });

  texture->map_read([&](const void* buffer,
                        size_t width,
                        size_t height,
                        size_t stride,
                        size_t channels,
                        const GLTexture::Type& type) {
    const T* buffer_data = (const T*)buffer;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        gl_log(GL_LOG, "%d, ", (int)buffer_data[channels * (y * stride + x) + 0]);
      }
      gl_log(GL_LOG, "\n");
    }
  });
  delete image;
  delete allocator;
  gl_log(GL_LOG, "...done with %s\n", __PRETTY_FUNCTION__);
}

void testOpenGL() {
  {
    // Test a bunch of different tiled convolutions
    std::vector<int> channels({3, 4, 6, 8, 12, 16, 32, 64, 128, 256, 512});

    for (const auto& input_channels : channels) {
      int tile_x = 1, tile_y = 1;
      squareFactors((input_channels + 3) / 4, tile_x, tile_y);

      for (const auto& output_channels : channels) {
        for (int size = 5; size < 8; size *= 2) {
          testOpenGLConv(1,
                         input_channels,
                         size,
                         size,
                         output_channels,
                         3,
                         3,
                         0,
                         1,
                         Conv,
                         0.1 * input_channels / 8,
                         true,
                         1,
                         1,
                         tile_x,
                         tile_y,
                         true);
        }

        for (int size = 5; size < 16; size *= 2) {
          testOpenGLConv(1,
                         input_channels,
                         size,
                         size,
                         output_channels,
                         3,
                         3,
                         0,
                         1,
                         ConvTranspose,
                         0.1 * input_channels / 8,
                         true,
                         1,
                         1,
                         tile_x,
                         tile_y,
                         true);
        }
      }
    }

    // Test various paddings and strides with tiled convolution
    for (int kernel_size = 1; kernel_size <= 5; kernel_size++) {
      for (int pad = 0; pad < kernel_size; pad++) {
        for (int stride = 1; stride <= 8; stride++) {
          testOpenGLConv(1,
                         16,
                         100,
                         100,
                         16,
                         kernel_size,
                         kernel_size,
                         pad,
                         stride,
                         Conv,
                         0.5,
                         true,
                         1,
                         1,
                         2,
                         2,
                         true);
        }

        for (int stride = 1; stride <= 8; stride++) {
          testOpenGLConv(1,
                         16,
                         100,
                         100,
                         16,
                         kernel_size,
                         kernel_size,
                         pad,
                         stride,
                         ConvTranspose,
                         0.5,
                         true,
                         1,
                         1,
                         2,
                         2,
                         true);
        }
      }
    }

    // Test a bunch of batched convolutions
    for (int kernel_size = 1; kernel_size <= 8; kernel_size++) {
      for (int stride = 1; stride <= 8; stride++) {
        testOpenGLConv(1,
                       16,
                       10,
                       10,
                       16,
                       kernel_size,
                       kernel_size,
                       0,
                       stride,
                       ConvTranspose,
                       0.5 * (1 + kernel_size / 3.0),
                       true,
                       1,
                       1);
      }

      for (int stride = 1; stride <= 8; stride++) {
        testOpenGLConv(1,
                       16,
                       10,
                       10,
                       16,
                       kernel_size,
                       kernel_size,
                       0,
                       stride,
                       Conv,
                       0.5 * (1 + kernel_size / 3.0),
                       true,
                       1,
                       1);
      }
    }
    for (const auto& channel : channels) {
      int tile_x = 1, tile_y = 1;
      squareFactors((channel + 3) / 4, tile_x, tile_y);
      // clang-format off
      testOpenGLConv(1, channel, 10, 10, channel, 3, 3, 0, 1, ConvPRelu, 0.1 * channel / 8, true, 1, 1, tile_x, tile_y, true);
      testOpenGLConv(1, channel, 10, 10, channel, 3, 3, 0, 1, ConvTransposePRelu, 0.1 * channel / 8, true, 1, 1, tile_x, tile_y, true);
      testOpenGLConv(1, channel, 10, 10, channel, 3, 3, 0, 1, ConvRelu, 0.1 * channel / 8, true, 1, 1, tile_x, tile_y, true);
      testOpenGLConv(1, channel, 10, 10, channel, 3, 3, 0, 1, ConvTransposeRelu, 0.1 * channel / 8, true, 1, 1, tile_x, tile_y, true);

      testOpenGLPRelu(1, channel, 13, 4, channel, tile_x, tile_y, 0.1);
      testOpenGLRelu(1, channel, 4, 17, tile_x, tile_y, 0.1);
      testOpenGLConv(1, channel, 16, 16, channel, 3, 3, 0, 2, MaxPool, 0.01, true, 1, 1, tile_x, tile_y, true);
      testOpenGLConv(1, channel, 16, 16, channel, 3, 3, 0, 2, AveragePool, 0.01, true, 1, 1, tile_x, tile_y, true);
      testOpenGLAdd(1, channel, 14, 8, 0.1, tile_x, tile_y);
      testOpenGLResize(1, channel, 16, 16, 2, 2, 0.1, tile_x, tile_y);
      // clang-format on
    }
  }

  {
    testGLTextureTypes<uint8_t>();
    testGLTextureTypes<float16_t>();

    testOpenGLCopyOps(1, 4, 4, 4, 1e-2);
    testOpenGLCopyOps(1, 3, 4, 4, 1e-2);
    testOpenGLCopyOps(1, 2, 4, 4, 1e-2);
    testOpenGLCopyOps(1, 1, 4, 4, 1e-2);
    testOpenGLCopyOps(1, 4, 2, 2, 1e-2);
    testOpenGLCopyOps(1, 4, 4, 4, 1e-2);
    testOpenGLCopyOps(1, 4, 1, 1, 1e-2);
    testOpenGLCopyOps(1, 4, 8, 8, 1e-2);
    testOpenGLCopyOps(1, 6, 8, 3, 1e-2);
    testOpenGLCopyOps(1, 4, 1, 2, 1e-2);
    testOpenGLCopyOps(1, 8, 6, 1, 1e-2);
    testOpenGLCopyOps(1, 8, 13, 18, 1e-2);
    testOpenGLCopyOps(1, 16, 13, 18, 1e-2);
    testOpenGLCopyOps(1, 13, 128, 90, 1e-2);
    testOpenGLCopyOps(1, 16, 1280, 720, 1e-2);

    testOpenGLCopyOps(1, 16, 4, 4, 1e-2, 2, 2);
    testOpenGLCopyOps(1, 64, 16, 16, 1e-2, 2, 2);
    testOpenGLCopyOps(1, 48, 13, 17, 1e-2, 3, 2);
    testOpenGLCopyOps(1, 512, 1, 1, 1e-2, 4, 16);
    testOpenGLCopyOps(1, 256, 7, 7, 1e-2, 8, 8);
    testOpenGLCopyOps(1, 20, 13, 17, 1e-2, 5, 1);

    // Test pooling operators
    LOG(INFO) << "Test pooling operators";
    testOpenGLConv(1, 4, 5, 5, 4, 3, 3, 0, 1, AveragePool, 0.01, true);
    testOpenGLConv(1, 4, 5, 5, 4, 5, 5, 0, 1, AveragePool, 0.5, true);

    testOpenGLConv(1, 4, 10, 10, 4, 3, 3, 0, 2, AveragePool, 0.01, true);
    testOpenGLConv(1, 4, 10, 10, 4, 3, 3, 1, 2, AveragePool, 0.01, true);
    testOpenGLConv(1, 4, 10, 10, 4, 3, 3, 2, 2, AveragePool, 0.01, true);

    testOpenGLConv(1, 4, 10, 10, 4, 3, 3, 0, 2, MaxPool, 0.01, true);
    testOpenGLConv(1, 4, 10, 10, 4, 3, 3, 1, 2, MaxPool, 0.01, true);
    testOpenGLConv(1, 4, 10, 10, 4, 3, 3, 2, 2, MaxPool, 0.01, true);

    // Test strided convolution
    LOG(INFO) << "Test strided convolution";
    testOpenGLConv(1, 4, 10, 10, 4, 3, 3, 0, 2, Conv, 0.5, true, 1, 1);
    testOpenGLConv(1, 4, 10, 10, 4, 3, 3, 1, 2, Conv, 0.5, true, 1, 1);
    testOpenGLConv(1, 4, 10, 10, 4, 3, 3, 2, 2, Conv, 0.5, true, 1, 1);

    testOpenGLConv(1, 4, 10, 10, 4, 3, 3, 0, 3, Conv, 0.5, true, 1, 1);
    testOpenGLConv(1, 4, 10, 10, 4, 3, 3, 1, 3, Conv, 0.5, true, 1, 1);
    testOpenGLConv(1, 4, 10, 10, 4, 3, 3, 2, 3, Conv, 0.5, true, 1, 1);

    // Test input batching
    LOG(INFO) << "Test input batching";
    testOpenGLConv(1, 4, 5, 5, 4, 3, 3, 0, 1, Conv, 0.5, false, 1, 1);
    testOpenGLConv(1, 8, 5, 5, 4, 3, 3, 0, 1, Conv, 0.5, false, 2, 1);
    testOpenGLConv(1, 12, 5, 5, 4, 3, 3, 0, 1, Conv, 0.5, false, 3, 1);
    testOpenGLConv(1, 16, 5, 5, 4, 3, 3, 0, 1, Conv, 0.5, false, 4, 1);

    testOpenGLConv(1, 4, 10, 10, 4, 3, 3, 0, 1, Conv, 1, true, 1, 1); // use random input
    testOpenGLConv(1, 8, 10, 10, 4, 3, 3, 0, 1, Conv, 1, true, 2, 1); // use random input
    testOpenGLConv(1, 12, 10, 10, 4, 3, 3, 0, 1, Conv, 2, true, 3, 1); // use random input
    testOpenGLConv(1, 16, 10, 10, 4, 3, 3, 0, 1, Conv, 2, true, 4, 1); // use random input
    testOpenGLConv(1, 32, 10, 10, 4, 3, 3, 0, 1, Conv, 4, true, 4, 1); // use random input

    // Test output batching
    LOG(INFO) << "Test output batching";
    testOpenGLConv(1, 4, 5, 5, 4, 3, 3, 0, 1, Conv, 0.5, false, 1, 1);
    testOpenGLConv(1, 4, 5, 5, 8, 3, 3, 0, 1, Conv, 0.5, false, 1, 2);
    testOpenGLConv(1, 4, 5, 5, 12, 3, 3, 0, 1, Conv, 0.5, false, 1, 3);
    testOpenGLConv(1, 4, 5, 5, 16, 3, 3, 0, 1, Conv, 0.5, false, 1, 4);

    testOpenGLConv(1, 4, 10, 10, 4, 3, 3, 0, 1, Conv, 0.5, true, 1, 1); // use random input
    testOpenGLConv(1, 4, 10, 10, 8, 3, 3, 0, 1, Conv, 1.5, true, 1, 2); // use random input
    testOpenGLConv(1, 4, 10, 10, 12, 3, 3, 0, 1, Conv, 0.5, true, 1, 3); // use random input
    testOpenGLConv(1, 4, 10, 10, 16, 3, 3, 0, 1, Conv, 0.5, true, 1, 4); // use random input

    // Test both
    LOG(INFO) << "Test both input and output batching";
    testOpenGLConv(1, 4, 5, 5, 4, 3, 3, 0, 1, Conv, 0.5, false, 1, 1);
    testOpenGLConv(1, 8, 5, 5, 8, 3, 3, 0, 1, Conv, 0.5, false, 2, 2);
    testOpenGLConv(1, 12, 5, 5, 12, 3, 3, 0, 1, Conv, 0.5, false, 3, 3);

    testOpenGLConv(1, 4, 10, 10, 4, 3, 3, 0, 1, Conv, 0.5, true, 1, 1); // use random input
    testOpenGLConv(1, 8, 10, 10, 8, 3, 3, 0, 1, Conv, 1, true, 2, 2); // use random input
    testOpenGLConv(1, 12, 10, 10, 12, 3, 3, 0, 1, Conv, 2, true, 3, 3); // use random input
    testOpenGLConv(1, 16, 10, 10, 16, 3, 3, 0, 1, Conv, 4, true, 4, 4); // use random input

    // Test different combination of batching
    LOG(INFO) << "Test mixed input and output batching sizes";
    testOpenGLConv(1, 16, 3, 3, 16, 3, 3, 0, 1, Conv, 4, false, 1, 2);
    testOpenGLConv(1, 16, 3, 3, 16, 3, 3, 0, 1, Conv, 4, false, 2, 2);
    testOpenGLConv(1, 16, 3, 3, 16, 3, 3, 0, 1, Conv, 4, false, 1, 4);
    testOpenGLConv(1, 16, 3, 3, 16, 3, 3, 0, 1, Conv, 4, false, 2, 4);

    testOpenGLConv(1, 16, 3, 3, 16, 3, 3, 0, 1, Conv, 4, false, 1, 1);
    testOpenGLConv(1, 16, 3, 3, 16, 3, 3, 0, 1, Conv, 4, false, 2, 1);
    testOpenGLConv(1, 16, 3, 3, 16, 3, 3, 0, 1, Conv, 4, false, 4, 1);
    testOpenGLConv(1, 16, 3, 3, 16, 3, 3, 0, 1, Conv, 4, false, 4, 2);

    testOpenGLConv(1, 16, 3, 3, 16, 3, 3, 0, 1, Conv, 4, true, 1, 1); // use random input
    testOpenGLConv(1, 16, 3, 3, 16, 3, 3, 0, 1, Conv, 4, true, 2, 1); // use random input
    testOpenGLConv(1, 16, 3, 3, 16, 3, 3, 0, 1, Conv, 4, true, 4, 1); // use random input
    testOpenGLConv(1, 16, 3, 3, 16, 3, 3, 0, 1, Conv, 4, true, 4, 2); // use random input

    testOpenGLConv(1, 16, 3, 3, 16, 3, 3, 0, 1, Conv, 4, true, 1, 1);
    testOpenGLConv(1, 16, 3, 3, 16, 3, 3, 0, 1, Conv, 4, true, 2, 1);
    testOpenGLConv(1, 16, 3, 3, 16, 3, 3, 0, 1, Conv, 4, true, 4, 1);
    testOpenGLConv(1, 16, 3, 3, 16, 3, 3, 0, 1, Conv, 4, true, 4, 2);

    testOpenGLConv(1, 16, 10, 10, 16, 3, 3, 0, 1, Conv, 4, true, 1, 1); // use random input
    testOpenGLConv(1, 16, 10, 10, 16, 3, 3, 0, 1, Conv, 4, true, 1, 2); // use random input
    testOpenGLConv(1, 16, 10, 10, 16, 3, 3, 0, 1, Conv, 4, true, 2, 1); // use random input
    testOpenGLConv(1, 16, 10, 10, 16, 3, 3, 0, 1, Conv, 4, true, 2, 2); // use random input
    testOpenGLConv(1, 16, 10, 10, 16, 3, 3, 0, 1, Conv, 4, true, 4, 1); // use random input
    testOpenGLConv(1, 16, 10, 10, 16, 3, 3, 0, 1, Conv, 4, true, 1, 4); // use random input

    // Test input/output channels
    for (int i = 0; i < 4; i++) {
      testOpenGLConv(1, 6, 10, 10, i, 3, 3, 0, 1, Conv, 4, true, 1, 1); // use random input
      testOpenGLConv(1, 6, 10, 10, i, 3, 3, 0, 1, Conv, 4, true, 2, 1); // use random input
    }

    // Test large input size
    LOG(INFO) << "Test large input size";
    testOpenGLConv(1, 4, 1280, 720, 4, 3, 3, 0, 1, Conv, 1, true, 1, 1); // use random input
    testOpenGLConv(1, 16, 1280, 720, 16, 3, 3, 0, 1, Conv, 4, true, 4, 4); // use random input

    // Test non standard input size
    testOpenGLConv(1, 16, 125, 73, 16, 3, 3, 0, 1, Conv, 4, true, 1, 1); // use random input
    testOpenGLConv(1, 16, 127, 71, 16, 3, 3, 0, 1, Conv, 4, true, 4, 4); // use random input

    // Test for different kernel size
    LOG(INFO) << "Test kernel sizes 4 to 6";
    for (int w = 4; w < 7; w++) {
      testOpenGLConv(1, 4, 128, 72, 4, w, w, 0, 1, Conv, 4 * (w / 3.0) * (w / 3.0), true, 1, 1);
    }

    // Test for random failures
    for (int i = 0; i < 10; i++) {
      testOpenGLConv(1, 6, 111, 111, 3, 3, 3, 0, 2, ConvTranspose, 0.5, true, 2, 1);
      testOpenGLConv(1, 16, 56, 56, 6, 4, 4, 0, 2, ConvTranspose, 0.5, true, 2, 2);
    }

    LOG(INFO) << "Test OpenGL ConvPRelu";
    testOpenGLConv(1, 16, 6, 6, 16, 3, 3, 0, 1, ConvPRelu, 2, true, 1, 1);
    testOpenGLConv(1, 4, 6, 6, 4, 3, 3, 0, 1, ConvPRelu, 1, true, 1, 1);
    testOpenGLConv(1, 8, 6, 6, 8, 3, 3, 0, 1, ConvPRelu, 2, true, 2, 2);
    testOpenGLConv(1, 16, 16, 16, 16, 3, 3, 0, 1, ConvPRelu, 4, true, 4, 4);
    testOpenGLConv(1, 12, 16, 16, 8, 3, 3, 0, 1, ConvPRelu, 4, true, 3, 1);
    testOpenGLConv(1, 16, 1280, 720, 16, 3, 3, 0, 1, ConvPRelu, 4, true, 4, 4);
    testOpenGLConv(1, 16, 1280, 720, 16, 3, 3, 0, 1, ConvPRelu, 4, true, 1, 1);

    LOG(INFO) << "Test OpenGL ConvTransposePRelu";
    testOpenGLConv(1, 16, 6, 6, 16, 3, 3, 0, 1, ConvTransposePRelu, 2, true, 1, 1);
    testOpenGLConv(1, 4, 6, 6, 4, 3, 3, 0, 1, ConvTransposePRelu, 1, true, 1, 1);
    testOpenGLConv(1, 8, 6, 6, 8, 3, 3, 0, 1, ConvTransposePRelu, 2, true, 2, 2);
    testOpenGLConv(1, 16, 16, 16, 16, 3, 3, 0, 1, ConvTransposePRelu, 4, true, 4, 4);
    testOpenGLConv(1, 12, 16, 16, 8, 3, 3, 0, 1, ConvTransposePRelu, 4, true, 3, 1);
    testOpenGLConv(1, 16, 1280, 720, 16, 3, 3, 0, 1, ConvTransposePRelu, 4, true, 4, 4);
    testOpenGLConv(1, 16, 1280, 720, 16, 3, 3, 0, 1, ConvTransposePRelu, 4, true, 1, 1);

    LOG(INFO) << "Test OpenGL ConvRelu";
    testOpenGLConv(1, 16, 6, 6, 16, 3, 3, 0, 1, ConvRelu, 2, true, 1, 1);
    testOpenGLConv(1, 4, 6, 6, 4, 3, 3, 0, 1, ConvRelu, 1, true, 1, 1);
    testOpenGLConv(1, 8, 6, 6, 8, 3, 3, 0, 1, ConvRelu, 2, true, 2, 2);
    testOpenGLConv(1, 16, 16, 16, 16, 3, 3, 0, 1, ConvRelu, 4, true, 4, 4);
    testOpenGLConv(1, 12, 16, 16, 8, 3, 3, 0, 1, ConvRelu, 4, true, 3, 1);
    testOpenGLConv(1, 16, 1280, 720, 16, 3, 3, 0, 1, ConvRelu, 4, true, 4, 4);
    testOpenGLConv(1, 16, 1280, 720, 16, 3, 3, 0, 1, ConvRelu, 4, true, 1, 1);

    LOG(INFO) << "Test OpenGL ConvTransposeRelu";
    testOpenGLConv(1, 16, 6, 6, 16, 3, 3, 0, 1, ConvTransposeRelu, 2, true, 1, 1);
    testOpenGLConv(1, 4, 6, 6, 4, 3, 3, 0, 1, ConvTransposeRelu, 1, true, 1, 1);
    testOpenGLConv(1, 8, 6, 6, 8, 3, 3, 0, 1, ConvTransposeRelu, 2, true, 2, 2);
    testOpenGLConv(1, 16, 16, 16, 16, 3, 3, 0, 1, ConvTransposeRelu, 4, true, 4, 4);
    testOpenGLConv(1, 12, 16, 16, 8, 3, 3, 0, 1, ConvTransposeRelu, 4, true, 3, 1);
    testOpenGLConv(1, 16, 1280, 720, 16, 3, 3, 0, 1, ConvTransposeRelu, 4, true, 4, 4);
    testOpenGLConv(1, 16, 1280, 720, 16, 3, 3, 0, 1, ConvTransposeRelu, 4, true, 1, 1);

    LOG(INFO) << "Test OpenGL PRelu";
    testOpenGLPRelu(1, 4, 16, 16, 4, 1, 1, 0.1);
    testOpenGLPRelu(1, 16, 16, 16, 1, 1, 1, 0.1);
    testOpenGLPRelu(1, 12, 16, 16, 1, 1, 1, 0.1);
    testOpenGLPRelu(1, 6, 640, 360, 6, 1, 1, 0.1);

    LOG(INFO) << "Test OpenGL Relu";
    testOpenGLRelu(1, 4, 16, 16, 1, 1, 0.1);
    testOpenGLRelu(1, 16, 16, 16, 1, 1, 0.1);
    testOpenGLRelu(1, 6, 640, 360, 1, 1, 0.1);

    LOG(INFO) << "Test OpenGL Add";
    testOpenGLAdd(1, 16, 640, 360, 0.1);
    testOpenGLAdd(1, 12, 640, 360, 0.1);

    LOG(INFO) << "Test OpenGL Sub";
    testOpenGLSub(1, 16, 640, 360, 0.1);
    testOpenGLSub(1, 12, 640, 360, 0.1);

    LOG(INFO) << "Test OpenGL Sigmoid";
    testOpenGLSigmoid(1, 4, 16, 16, 0.1);
    testOpenGLSigmoid(1, 12, 64, 48, 0.1);
    testOpenGLSigmoid(1, 6, 640, 360, 0.1);

    LOG(INFO) << "Test OpenGL Tanh";
    testOpenGLTanh(1, 4, 16, 16, 0.1);
    testOpenGLTanh(1, 12, 64, 48, 0.1);
    testOpenGLTanh(1, 6, 640, 360, 0.1);

    LOG(INFO) << "Test OpenGL Mul";
    testOpenGLMul(1, 4, 16, 16, 0.1);
    testOpenGLMul(1, 12, 64, 48, 0.1);
    testOpenGLMul(1, 6, 640, 360, 0.1);

    LOG(INFO) << "Test OpenGL Concat";
    testOpenGLConcat(1, std::vector<int>{4, 4}, 16, 16);
    testOpenGLConcat(1, std::vector<int>{4, 4, 4}, 16, 16);
    testOpenGLConcat(1, std::vector<int>{4, 4, 4, 4}, 16, 16);
    testOpenGLConcat(1, std::vector<int>{8, 4, 12}, 16, 16);
    testOpenGLConcat(1, std::vector<int>{12, 16, 8}, 16, 16);
    testOpenGLConcat(1, std::vector<int>{60, 24, 36}, 16, 16);

    testOpenGLConcat(1, std::vector<int>{12, 16, 8}, 16, 16, true);
    testOpenGLConcat(1, std::vector<int>{60, 24, 36}, 16, 16, true);

    LOG(INFO) << "Test OpenGL Softmax";
    testOpenGLSoftmax(1, 100, 0.1);
    testOpenGLSoftmax(1, 500, 0.1);
    testOpenGLSoftmax(1, 1000, 0.1);
    testOpenGLSoftmax(1, 5000, 0.1);

    LOG(INFO) << "Test OpenGL InstanceNorm";
    testOpenGLInstanceNorm(1, 4, 16, 16, 0.2);
    testOpenGLInstanceNorm(1, 4, 20, 20, 0.2);
    testOpenGLInstanceNorm(1, 4, 128, 128, 0.2);
    testOpenGLInstanceNorm(1, 12, 120, 140, 0.3);
    testOpenGLInstanceNorm(1, 3, 120, 140, 0.2);
    testOpenGLInstanceNorm(1, 4, 192, 192, 0.2);

    testOpenGLInstanceNorm(1, 4, 258, 198, 0.2);
    testOpenGLInstanceNorm(1, 8, 338, 198, 0.2);
    testOpenGLInstanceNorm(1, 12, 334, 194, 0.2);
    testOpenGLInstanceNorm(1, 16, 324, 184, 0.2);
    testOpenGLInstanceNorm(1, 6, 640, 360, 0.2);

    LOG(INFO) << "Test OpenGL InstanceNormPRelu";
    testOpenGLInstanceNormPRelu(1, 4, 16, 16, 0.2);
    testOpenGLInstanceNormPRelu(1, 4, 20, 20, 0.2);
    testOpenGLInstanceNormPRelu(1, 4, 128, 128, 0.2);
    testOpenGLInstanceNormPRelu(1, 12, 120, 140, 0.3);
    testOpenGLInstanceNormPRelu(1, 3, 120, 140, 0.2);
    testOpenGLInstanceNormPRelu(1, 4, 192, 192, 0.2);

    testOpenGLInstanceNormPRelu(1, 4, 258, 198, 0.2);
    testOpenGLInstanceNormPRelu(1, 8, 338, 198, 0.2);
    testOpenGLInstanceNormPRelu(1, 12, 334, 194, 0.2);
    testOpenGLInstanceNormPRelu(1, 16, 324, 184, 0.2);
    testOpenGLInstanceNormPRelu(1, 6, 640, 360, 0.2);

    LOG(INFO) << "Test OpenGL ResizeNearest";
    testOpenGLResize(1, 4, 16, 16, 1, 1, 0.1);
    testOpenGLResize(1, 4, 16, 16, 2, 2, 0.1);
    testOpenGLResize(1, 4, 16, 16, 3, 3, 0.1);
    testOpenGLResize(1, 4, 16, 16, 4, 4, 0.1);
    testOpenGLResize(1, 16, 25, 25, 3, 3, 0.1);
    testOpenGLResize(1, 16, 25, 25, 3, 3, 0.1);
    testOpenGLResize(1, 12, 25, 25, 3, 3, 0.1);
    testOpenGLResize(1, 4, 720, 1280, 3, 3, 0.1);

    // debug style transfer
    // conv
    testOpenGLConv(1, 3, 82, 82, 8, 9, 9, 0, 1, Conv, 4, true, 1, 1);
    testOpenGLConv(1, 8, 74, 74, 8, 3, 3, 0, 1, Conv, 4, true, 1, 1);
    testOpenGLConv(1, 8, 82, 82, 12, 3, 3, 0, 1, Conv, 4, true, 1, 1);
    testOpenGLConv(1, 12, 82, 82, 12, 3, 3, 0, 1, Conv, 4, true, 1, 1);

    // convtranspose
    testOpenGLConv(1, 16, 56, 56, 6, 4, 4, 0, 2, ConvTranspose, 0.5, true, 2, 2);
    testOpenGLConv(1, 6, 112, 112, 3, 4, 4, 0, 2, ConvTranspose, 0.5, true, 2, 1);

    LOG(INFO) << "Test OpenGL PadImage";
    testOpenGLPadImage(1, 3, 11, 11, 0, 1, 0, 1, 0.001);
    testOpenGLPadImage(1, 3, 50, 80, 0, 1, 0, 1, 0.001);
    testOpenGLPadImage(1, 12, 50, 80, 10, 9, 10, 9, 0.001);

    LOG(INFO) << "Test OpenGL Preprocess";
    testOpenGLPreprocess(1, 4, 8, 8, 0.20);
    testOpenGLPreprocess(1, 4, 1280, 720, 0.20);

    LOG(INFO) << "Test OpenGL Deprocess";
    testOpenGLDeprocess(1, 3, 8, 8, 0.01);
    testOpenGLDeprocess(1, 3, 1280, 720, 0.01);

    LOG(INFO) << "Test OpenGL NormalizePlanarYUV";
    testOpenGLNormPlanarYUV(1, 3, 8, 8, 0.01);
    testOpenGLNormPlanarYUV(1, 3, 192, 192, 0.01);

    //  for (int i = 0; i < 4; i += 1) {
    //    LOG(INFO) << "C: " << 4 << ", H: " << 1280 + i << ", W: " << 720 + i;
    //    OpenGL_copyops_speedtest(1, 4, 1280, 720 + i, 4, 3, 3, 0, 0.5);
    //  }

    //  for (int i = 0; i < 1; i += 1) {
    //    LOG(INFO) << "C: " << 16 << ", H: " << 1280 + i << ", W: " << 720 + i;
    //    OpenGL_copyops_speedtest(1, 16, 1280, 720 + i, 16, 3, 3, 0, 0.5);
    //  }
    //
    //  for (int i = 0; i < 9; i += 1) {
    //    LOG(INFO) << "C: " << 16 << ", H: " << 1280 + i << ", W: " << 720 + i;
    //    OpenGL_speedtest(1, 16, 1280, 720 + i, 16, 3, 3, 0, 0.5);
    //  }

    // Multi-Batch Tests
    LOG(INFO) << "Test OpenGL Multi-batch Support";
    testOpenGLCopyOps(2, 4, 4, 4, 1e-2);
    testOpenGLCopyOps(3, 4, 4, 4, 1e-2);
    testOpenGLCopyOps(5, 4, 4, 4, 1e-2);
    testOpenGLConv(2, 4, 5, 5, 4, 3, 3, 0, 1, AveragePool, 0.01, true);
    testOpenGLConv(2, 4, 10, 10, 4, 3, 3, 0, 2, MaxPool, 0.01, true);
    testOpenGLConv(3, 4, 10, 10, 4, 3, 3, 0, 2, Conv, 0.5, true, 1, 1);
    testOpenGLConv(5, 4, 10, 10, 4, 3, 3, 0, 2, Conv, 0.5, true, 1, 1);
    testOpenGLConv(7, 4, 10, 10, 4, 3, 3, 0, 2, Conv, 0.5, true, 1, 1);
    testOpenGLConv(11, 4, 10, 10, 4, 3, 3, 0, 2, Conv, 0.5, true, 1, 1);
    testOpenGLConv(12, 4, 10, 10, 4, 3, 3, 0, 2, Conv, 0.5, true, 1, 1);
    testOpenGLConv(21, 4, 10, 10, 4, 3, 3, 0, 2, Conv, 0.5, true, 1, 1);
    testOpenGLConv(50, 4, 10, 10, 4, 3, 3, 0, 2, Conv, 0.5, true, 1, 1);
    testOpenGLConv(3, 4, 10, 10, 4, 3, 3, 0, 2, ConvTranspose, 0.5, true, 1, 1);
    testOpenGLConv(3, 16, 6, 6, 16, 3, 3, 0, 1, ConvPRelu, 2, true, 1, 1);
    testOpenGLConv(3, 16, 6, 6, 16, 3, 3, 0, 1, ConvTransposePRelu, 2, true, 1, 1);

    testOpenGLPRelu(3, 4, 16, 16, 4, 1, 1, 0.1);
    testOpenGLPRelu(5, 4, 16, 16, 4, 1, 1, 0.1);

    testOpenGLRelu(3, 4, 16, 16, 1, 1, 0.1);
    testOpenGLRelu(7, 4, 16, 16, 1, 1, 0.1);

    testOpenGLAdd(3, 16, 640, 360, 0.1);
    testOpenGLAdd(9, 16, 640, 360, 0.1);

    testOpenGLSigmoid(3, 4, 16, 16, 0.1);
    testOpenGLSigmoid(11, 4, 16, 16, 0.1);

    testOpenGLInstanceNorm(3, 4, 16, 16, 0.2);
    testOpenGLInstanceNorm(13, 4, 16, 16, 0.2);

    testOpenGLInstanceNormPRelu(3, 4, 16, 16, 0.2);
    testOpenGLInstanceNormPRelu(15, 4, 16, 16, 0.2);

    testOpenGLResize(3, 4, 16, 16, 1, 1, 0.1);
    testOpenGLResize(16, 4, 16, 16, 1, 1, 0.1);

    testOpenGLPadImage(3, 3, 4, 4, 0, 1, 0, 1, 0.01);
    testOpenGLPadImage(23, 3, 4, 4, 0, 1, 0, 1, 0.01);

    testOpenGLSoftmax(3, 1000, 0.1);
    testOpenGLSoftmax(27, 100, 0.1);

    testOpenGLNormPlanarYUV(4, 3, 192, 192, 0.01);

    // Test Tiling
    testOpenGLSoftmax(3, 1000, 0.1, true);
    testOpenGLSoftmax(9, 523, 0.1, true);
    testOpenGLSoftmax(27, 100, 0.1, true);
  }

  LOG(INFO) << "End of OpenGL tests";
}
} // namespace caffe2
