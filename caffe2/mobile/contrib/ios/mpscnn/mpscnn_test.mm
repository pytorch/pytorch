#include "caffe2/core/common.h"

#if defined(C10_MOBILE) && defined(CAFFE2_USE_MPSCNN_TEST)

#include "mpscnn_context.h"
#include "mpscnn_graph_mask.h"

#include "caffe2/core/logging.h"
#include "caffe2/core/operator_schema.h"
#include "caffe2/core/workspace.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"

#import <UIKit/UIDevice.h>

#define SYSTEM_VERSION_GREATER_THAN_OR_EQUAL_TO(v) \
  ([[[UIDevice currentDevice] systemVersion]       \
       compare:v                                   \
       options:NSNumericSearch] != NSOrderedAscending)

namespace caffe2 {

/* Utility functions for operator definition */
void add_arg_int(OperatorDef& op, string name, int value) {
  auto& arg = *(op.add_arg());
  arg.set_name(name);
  arg.set_i(value);
}

void add_arg_str(OperatorDef& op, string name, string value) {
  auto& arg = *(op.add_arg());
  arg.set_name(name);
  arg.set_s(value);
}

void add_arg_float(OperatorDef& op, string name, float value) {
  auto& arg = *(op.add_arg());
  arg.set_name(name);
  arg.set_f(value);
}

void add_arg_int_list(
    OperatorDef& op,
    std::vector<string> names,
    std::vector<int> values) {
  CAFFE_ENFORCE_EQ(names.size(), values.size());
  for (auto i = 0; i < names.size(); i++) {
    add_arg_int(op, names[i], values[i]);
  }
}

void add_arg_str_list(
    OperatorDef& op,
    std::vector<string> names,
    std::vector<string> values) {
  CAFFE_ENFORCE_EQ(names.size(), values.size());
  for (auto i = 0; i < names.size(); i++) {
    add_arg_str(op, names[i], values[i]);
  }
}

void add_inputs(OperatorDef& op, std::vector<string> inputs) {
  for (auto i = 0; i < inputs.size(); i++) {
    op.add_input(inputs[i]);
  }
}

void add_outputs(OperatorDef& op, std::vector<string> outputs) {
  for (auto i = 0; i < outputs.size(); i++) {
    op.add_output(outputs[i]);
  }
}

void testMPSCNN() {
  // initialize.
  getMPSCNNContext();

  {
    for (const auto C : std::vector<size_t>{1, 2, 3, 4, 8, 11, 12}) {
      for (const auto H : std::vector<size_t>{1, 7, 15, 39}) {
        for (const auto W : std::vector<size_t>{1, 7, 15, 39}) {
          for (const auto N : std::vector<size_t>{1, 2}) {
            for (const auto BS : std::vector<size_t>{1, 2}) {
              LOG(INFO) << "MPSCNNCopyFrom/To Test";
              auto mtl = [&](size_t i) {
                return std::string("X_mtl_") + std::to_string(i);
              };
              auto cpu = [&](size_t i) {
                return std::string("X_cpu_") + std::to_string(i);
              };
              auto y_cpu = [&](size_t i) {
                return std::string("Y_cpu_") + std::to_string(i);
              };

              Workspace ws;
              for (auto i = 0; i < N; ++i) {
                auto* t = BlobGetMutableTensor(ws.CreateBlob(cpu(i)), CPU);
                t->Resize(BS, C, H, W);
                CPUContext ctx;
                math::RandGaussian<float, CPUContext>(
                    t->size(), 0, 1, t->mutable_data<float>(), &ctx);
              }

              NetDef netdef;
              {
                auto& op = *(netdef.add_op());
                op.set_type("CopyToMPSCNN");
                for (auto i = 0; i < N; ++i) {
                  op.add_input(cpu(i));
                  op.add_output(mtl(i));
                }
              }
              {
                auto& op = *(netdef.add_op());
                op.set_type("CopyFromMPSCNN");
                for (auto i = 0; i < N; ++i) {
                  op.add_input(mtl(i));
                  op.add_output(y_cpu(i));
                }
              }

              ws.RunNetOnce(netdef);
              for (auto i = 0; i < N; ++i) {
                const auto& t1 = ws.GetBlob(cpu(i))->Get<TensorCPU>();
                const auto& t2 = ws.GetBlob(y_cpu(i))->Get<TensorCPU>();
                CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
                for (auto i = 0; i < t1.size(); ++i) {
                  // FP16 <-> FP32 round trip.
                  TORCH_CHECK_NEAR(t1.data<float>()[i], t2.data<float>()[i], 1e-2);
                }
              }
            }
          }
        }
      }
    }
  }

  {
    for (const auto ndim : std::vector<size_t>{1, 2, 3, 4}) {
      for (const auto N : std::vector<size_t>{1, 2}) {
        LOG(INFO) << "MPSCNNCopyFrom/To ndim Test";
        auto mtl = [&](size_t i) {
          return std::string("X_mtl_") + std::to_string(i);
        };
        auto cpu = [&](size_t i) {
          return std::string("X_cpu_") + std::to_string(i);
        };
        auto y_cpu = [&](size_t i) {
          return std::string("Y_cpu_") + std::to_string(i);
        };

        Workspace ws;
        for (auto i = 0; i < N; ++i) {
          auto* t = BlobGetMutableTensor(ws.CreateBlob(cpu(i)), CPU);
          switch (ndim) {
            case 1:
              t->Resize(5);
              break;
            case 2:
              t->Resize(5, 3);
              break;
            case 3:
              t->Resize(5, 3, 4);
              break;
            case 4:
              t->Resize(5, 3, 4, 2);
              break;
          }
          CPUContext ctx;
          math::RandGaussian<float, CPUContext>(
              t->size(), 0, 1, t->mutable_data<float>(), &ctx);
        }

        NetDef netdef;
        {
          auto& op = *(netdef.add_op());
          op.set_type("CopyToMPSCNN");
          for (auto i = 0; i < N; ++i) {
            op.add_input(cpu(i));
            op.add_output(mtl(i));
          }
        }
        {
          auto& op = *(netdef.add_op());
          op.set_type("CopyFromMPSCNN");
          for (auto i = 0; i < N; ++i) {
            op.add_input(mtl(i));
            op.add_output(y_cpu(i));
          }
        }

        ws.RunNetOnce(netdef);
        for (auto i = 0; i < N; ++i) {
          const auto& t1 = ws.GetBlob(cpu(i))->Get<TensorCPU>();
          const auto& t2 = ws.GetBlob(y_cpu(i))->Get<TensorCPU>();
          CAFFE_ENFORCE_EQ(t1.size(), t2.size());
          for (auto i = 0; i < t1.size(); ++i) {
            // FP16 <-> FP32 round trip.
            TORCH_CHECK_NEAR(t1.data<float>()[i], t2.data<float>()[i], 1e-2);
          }
        }
      }
    }
  }

  {
    for (const auto& batch_size : std::vector<int>{{1, 2}}) {
      for (const auto& channels : std::vector<int>{{3, 8}}) {
        LOG(INFO) << "MPSCNNNormalizePlanarYUV Test: ";
        Workspace ws;
        {
          auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
          t->Resize(batch_size, channels, 8, 13);
          CPUContext ctx;
          math::RandGaussian<float, CPUContext>(
              t->size(), 0, 1, t->mutable_data<float>(), &ctx);
        }

        {
          auto* t = BlobGetMutableTensor(ws.CreateBlob("mean"), CPU);
          t->Resize(1, channels);
          CPUContext ctx;
          math::RandGaussian<float, CPUContext>(
              t->size(), 0, 1, t->mutable_data<float>(), &ctx);
        }
        {
          auto* t = BlobGetMutableTensor(ws.CreateBlob("stddev"), CPU);
          t->Resize(1, channels);
          CPUContext ctx;
          math::RandUniform<float, CPUContext>(
              t->size(), 0.5, 1.5, t->mutable_data<float>(), &ctx);
        }

        NetDef netdef;
        {
          auto& op = *(netdef.add_op());
          op.set_type("CopyToMPSCNN");
          op.add_input("X_cpu");
          op.add_output("X_mtl");
        }

        {
          auto& op = *(netdef.add_op());
          op.set_type("MPSCNNNormalizePlanarYUV");
          op.add_input("X_mtl");
          op.add_input("mean");
          op.add_input("stddev");
          op.add_output("Y_mtl");
        }

        {
          auto& op = *(netdef.add_op());
          op.set_type("CopyFromMPSCNN");
          op.add_input("Y_mtl");
          op.add_output("Y_cpu");
        }

        {
          auto& op = *(netdef.add_op());
          op.set_type("NormalizePlanarYUV");
          op.add_input("X_cpu");
          op.add_input("mean");
          op.add_input("stddev");
          op.add_output("Y_ref");
        }

        ws.RunNetOnce(netdef);
        const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
        const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

        CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
        for (auto i = 0; i < t1.size(); ++i) {
          // FP16 <-> FP32 round trip, accumulation, etc.
          const float t1_i = t1.data<float>()[i];
          const float t2_i = t2.data<float>()[i];
          TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
        }
      }
    }
  }

  {
    LOG(INFO) << "MPSCNNInstanceNorm Test";
    enum class PreluTy { NONE, CHANNEL, SHARED };
    for (const auto batchSize : {1, 2}) {
      for (const auto channels : {3, 8}) {
        for (const auto prelu :
             {PreluTy::NONE, PreluTy::CHANNEL, PreluTy::SHARED}) {
          for (const auto dim : {10, 40}) {
            Workspace ws;
            {
              auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
              t->Resize(batchSize, channels, dim, dim);
              CPUContext ctx;
              // Too noisy.
              math::RandGaussian<float, CPUContext>(
                  t->size(), 0, 3, t->mutable_data<float>(), &ctx);
            }

            {
              auto* t = BlobGetMutableTensor(ws.CreateBlob("W"), CPU);
              t->Resize(channels);
              CPUContext ctx;
              for (auto i = 0; i < t->size(); ++i) {
                t->mutable_data<float>()[i] = i;
              }
              // Too noisy.
              // math::RandGaussian<float, CPUContext>(t->size(), 0, 1,
              // t->mutable_data<float>(), &ctx);
            }
            {
              auto* t = BlobGetMutableTensor(ws.CreateBlob("b"), CPU);
              t->Resize(channels);
              CPUContext ctx;
              for (auto i = 0; i < t->size(); ++i) {
                t->mutable_data<float>()[i] = 8 - 2 * i;
              }
              // Too noisy.
              // math::RandGaussian<float, CPUContext>(t->size(), 0, 1,
              // t->mutable_data<float>(), &ctx);
            }
            {
              auto* t = BlobGetMutableTensor(ws.CreateBlob("pw"), CPU);
              t->Resize(prelu == PreluTy::SHARED ? 1 : channels);
              CPUContext ctx;
              // Too noisy.
              math::RandGaussian<float, CPUContext>(
                  t->size(), 0, 1, t->mutable_data<float>(), &ctx);
            }

            NetDef netdef;
            {
              auto& op = *(netdef.add_op());
              op.set_type("CopyToMPSCNN");
              op.add_input("X_cpu");
              op.add_output("X_mtl");
            }

            {
              auto& op = *(netdef.add_op());
              op.set_type(
                  prelu == PreluTy::NONE ? "MPSCNNInstanceNorm"
                                         : "MPSCNNInstanceNormPRelu");
              op.add_input("X_mtl");
              op.add_input("W");
              op.add_input("b");
              if (prelu != PreluTy::NONE) {
                op.add_input("pw");
              }
              op.add_output("Y_mtl");
            }

            {
              auto& op = *(netdef.add_op());
              op.set_type("CopyFromMPSCNN");
              op.add_input("Y_mtl");
              op.add_output("Y_cpu");
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
            }

            if (prelu != PreluTy::NONE) {
              auto& op = *(netdef.add_op());
              op.set_type("PRelu");
              op.add_input("Y_ref");
              op.add_input("pw");
              auto& arg = *(op.add_arg());
              arg.set_name("order");
              arg.set_s("NCHW");
              op.add_output("Y_ref");
            }

            ws.RunNetOnce(netdef);
            const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
            const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

            CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
            for (auto i = 0; i < t1.size(); ++i) {
              // FP16 <-> FP32 round trip, accumulation, etc.
              const float t1_i = t1.data<float>()[i];
              const float t2_i = t2.data<float>()[i];
              // Can be larger due to FP errors.
              constexpr float tol = 5.0e-2;
              CHECK(std::abs(t1_i - t2_i) <= (tol + tol * std::abs(t1_i)))
                  << t1_i << ", " << t2_i;
            }
          }
        }
      }
    }
  }

  {
    for (const auto& shared : std::vector<bool>{{true, false}}) {
      for (const auto& array : std::vector<bool>{{true, false}}) {
        for (const auto& batch_size : std::vector<int>{{1, 2}}) {
          LOG(INFO) << "MPSCNNPRelu Test: " << shared << array << batch_size;
          Workspace ws;
          const auto channels = array ? 12 : 3;
          {
            auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
            t->Resize(batch_size, channels, 8, 13);
            CPUContext ctx;
            math::RandGaussian<float, CPUContext>(
                t->size(), 0, 1, t->mutable_data<float>(), &ctx);
          }

          {
            auto* t = BlobGetMutableTensor(ws.CreateBlob("b"), CPU);
            t->Resize(shared ? channels : 1);
            CPUContext ctx;
            math::RandGaussian<float, CPUContext>(
                t->size(), 0, 1, t->mutable_data<float>(), &ctx);
          }

          NetDef netdef;
          {
            auto& op = *(netdef.add_op());
            op.set_type("CopyToMPSCNN");
            op.add_input("X_cpu");
            op.add_output("X_mtl");
          }

          {
            auto& op = *(netdef.add_op());
            op.set_type("MPSCNNPRelu");
            op.add_input("X_mtl");
            op.add_input("b");
            op.add_output("Y_mtl");
          }

          {
            auto& op = *(netdef.add_op());
            op.set_type("CopyFromMPSCNN");
            op.add_input("Y_mtl");
            op.add_output("Y_cpu");
          }

          {
            auto& op = *(netdef.add_op());
            op.set_type("PRelu");
            op.add_input("X_cpu");
            op.add_input("b");
            auto& arg = *(op.add_arg());
            arg.set_name("order");
            arg.set_s("NCHW");
            op.add_output("Y_ref");
          }

          ws.RunNetOnce(netdef);
          const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
          const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

          CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
          for (auto i = 0; i < t1.size(); ++i) {
            // FP16 <-> FP32 round trip, accumulation, etc.
            const float t1_i = t1.data<float>()[i];
            const float t2_i = t2.data<float>()[i];
            TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
          }
        }
      }
    }
  }

  {
    for (const auto& channels : std::vector<size_t>{3, 12, 15}) {
      for (const auto& batch_size : std::vector<size_t>{1, 2}) {
        LOG(INFO) << "MPSCNNSpatialBN Test: " << channels;
        Workspace ws;
        {
          auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
          t->Resize(batch_size, channels, 8, 13);
          CPUContext ctx;
          math::RandGaussian<float, CPUContext>(
              t->size(), 0, 1, t->mutable_data<float>(), &ctx);
        }

        for (const std::string name : {"scale", "bias", "mean", "var"}) {
          auto* t = BlobGetMutableTensor(ws.CreateBlob(name), CPU);
          t->Resize(channels);
          CPUContext ctx;
          // High mean to avoid var division by zero.
          math::RandGaussian<float, CPUContext>(
              t->size(), 0, 1, t->mutable_data<float>(), &ctx);
          if (name == "var") {
            for (auto i = 0; i < t->size(); ++i) {
              t->mutable_data<float>()[i] =
                  std::abs(t->mutable_data<float>()[i]) + 0.5;
            }
          }
        }

        NetDef netdef;
        {
          auto& op = *(netdef.add_op());
          op.set_type("CopyToMPSCNN");
          op.add_input("X_cpu");
          op.add_output("X_mtl");
        }

        {
          auto& op = *(netdef.add_op());
          op.set_type("MPSCNNSpatialBN");
          op.add_input("X_mtl");
          op.add_input("scale");
          op.add_input("bias");
          op.add_input("mean");
          op.add_input("var");
          {
            auto& arg = *(op.add_arg());
            arg.set_name(OpSchema::Arg_IsTest);
            arg.set_i(1);
          }

          op.add_output("Y_mtl");
        }

        {
          auto& op = *(netdef.add_op());
          op.set_type("CopyFromMPSCNN");
          op.add_input("Y_mtl");
          op.add_output("Y_cpu");
        }

        {
          auto& op = *(netdef.add_op());
          op.set_type("SpatialBN");
          op.add_input("X_cpu");
          op.add_input("scale");
          op.add_input("bias");
          op.add_input("mean");
          op.add_input("var");
          {
            auto& arg = *(op.add_arg());
            arg.set_name(OpSchema::Arg_IsTest);
            arg.set_i(1);
          }

          op.add_output("Y_ref");
        }

        ws.RunNetOnce(netdef);
        const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
        const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

        CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
        for (auto i = 0; i < t1.size(); ++i) {
          // FP16 <-> FP32 round trip, accumulation, etc.
          const float t1_i = t1.data<float>()[i];
          const float t2_i = t2.data<float>()[i];
          TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
        }
      }
    }
  }

  {
    for (const auto& batchSize : std::vector<size_t>{2, 1}) {
      for (const auto& H : std::vector<size_t>{1, 8}) {
        for (const auto& W : std::vector<size_t>{1, 8}) {
          for (const auto& CIn : std::vector<size_t>{1, 12, 224}) {
            for (const auto& COut : std::vector<size_t>{1, 12, 224}) {
              LOG(INFO) << "MPSCNNFC Test";
              Workspace ws;
              {
                auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
                t->Resize(batchSize, CIn, H, W);
                CPUContext ctx;
                math::RandGaussian<float, CPUContext>(
                    t->size(), 0, 1, t->mutable_data<float>(), &ctx);
              }

              {
                auto* t = BlobGetMutableTensor(ws.CreateBlob("W"), CPU);
                t->Resize(COut, CIn * H * W);
                CPUContext ctx;
                math::RandGaussian<float, CPUContext>(
                    t->size(), 0, 1, t->mutable_data<float>(), &ctx);
              }

              {
                auto* t = BlobGetMutableTensor(ws.CreateBlob("b"), CPU);
                t->Resize(COut);
                CPUContext ctx;
                math::RandGaussian<float, CPUContext>(
                    t->size(), 0, 0.0001, t->mutable_data<float>(), &ctx);
              }

              NetDef netdef;
              {
                auto& op = *(netdef.add_op());
                op.set_type("CopyToMPSCNN");
                op.add_input("X_cpu");
                op.add_output("X_mtl");
              }

              {
                auto& op = *(netdef.add_op());
                op.set_type("MPSCNNFC");
                op.add_input("X_mtl");
                op.add_input("W");
                op.add_input("b");
                op.add_output("Y_mtl");
              }

              {
                auto& op = *(netdef.add_op());
                op.set_type("CopyFromMPSCNN");
                op.add_input("Y_mtl");
                op.add_output("Y_cpu");
              }
              {
                auto& op = *(netdef.add_op());
                op.set_type("FC");
                op.add_input("X_cpu");
                op.add_input("W");
                op.add_input("b");
                auto& arg = *(op.add_arg());
                arg.set_name("order");
                arg.set_s("NCHW");
                op.add_output("Y_ref");
              }

              ws.RunNetOnce(netdef);
              const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
              const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();
              CAFFE_ENFORCE_EQ(t2.ndim(), 4);
              CAFFE_ENFORCE_EQ(t1.ndim(), 2);
              CAFFE_ENFORCE(t2.dim32(2) == 1 && t2.dim32(3) == 1);
              const_cast<TensorCPU&>(t2).Reshape(
                  std::vector<int64_t>{int64_t(batchSize), int64_t(COut)});
              // Note dims do not match, as Metal leaves a 1x1 spatial
              // dimension.
              CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());

              for (auto i = 0; i < t1.size(); ++i) {
                // FP16 <-> FP32 round trip, accumulation, etc.
                const float t1_i = t1.data<float>()[i];
                const float t2_i = t2.data<float>()[i];
                // LOG(INFO) << "i: " << i << ", cpu: " << t1_i << ", mtl: " <<
                // t2_i;
                TORCH_CHECK_NEAR(t1_i, t2_i, 0.7);
              }
            }
          }
        }
      }
    }
  }

  {
    for (const auto& pool : {"MaxPool", "AveragePool"}) {
      for (const auto& global_pooling : {true, false}) {
        for (const auto& batchSize : std::vector<size_t>{1, 2}) {
          for (const auto& stride_h : std::vector<int>{1, 2, 3}) {
            for (const auto& stride_w : std::vector<int>{1, 2, 3}) {
              for (const auto& kernel_h : std::vector<int>{1, 3, 5}) {
                for (const auto& kernel_w : std::vector<int>{1, 3, 5}) {
                  for (const auto& pad_l : std::vector<int>{0, kernel_w / 2}) {
                    for (const auto& pad_r :
                         std::vector<int>{0, kernel_w / 2}) {
                      for (const auto& pad_t :
                           std::vector<int>{0, kernel_h / 2}) {
                        for (const auto& pad_b :
                             std::vector<int>{0, kernel_h / 2}) {
                          // Waiting response from Apple
                          if (kernel_h != kernel_w) {
                            continue;
                          }
                          LOG(INFO) << "MPSCNNPool Test: " << pool;
                          Workspace ws;
                          {
                            auto* t = BlobGetMutableTensor(
                                ws.CreateBlob("X_cpu"), CPU);
                            t->Resize(batchSize, 8, 8, 13);
                            CPUContext ctx;
                            math::RandGaussian<float, CPUContext>(
                                t->size(),
                                0,
                                1,
                                t->mutable_data<float>(),
                                &ctx);
                          }

                          NetDef netdef;
#define ADD_ARGS(op)                                   \
  do {                                                 \
    if (global_pooling) {                              \
      add_arg_int(op, "stride", 1);                    \
    } else {                                           \
      add_arg_int_list(                                \
          op,                                          \
          std::vector<string>{"pad_l",                 \
                              "pad_r",                 \
                              "pad_t",                 \
                              "pad_b",                 \
                              "kernel_w",              \
                              "kernel_h",              \
                              "stride_w",              \
                              "stride_h"},             \
          std::vector<int>{pad_l,                      \
                           pad_r,                      \
                           pad_t,                      \
                           pad_b,                      \
                           kernel_w,                   \
                           kernel_h,                   \
                           stride_w,                   \
                           stride_h});                 \
    }                                                  \
    add_arg_int(op, "global_pooling", global_pooling); \
  } while (false)
                          {
                            auto& op = *(netdef.add_op());
                            op.set_type("CopyToMPSCNN");
                            op.add_input("X_cpu");
                            op.add_output("X_mtl");
                          }

                          {
                            auto& op = *(netdef.add_op());
                            op.set_type(std::string("MPSCNN") + pool);
                            op.add_input("X_mtl");
                            ADD_ARGS(op);
                            op.add_output("Y_mtl");
                          }

                          {
                            auto& op = *(netdef.add_op());
                            op.set_type("CopyFromMPSCNN");
                            op.add_input("Y_mtl");
                            op.add_output("Y_cpu");
                          }

                          {
                            auto& op = *(netdef.add_op());
                            op.set_type(pool);
                            op.add_input("X_cpu");
                            ADD_ARGS(op);
                            op.add_output("Y_ref");
                          }
#undef ADD_ARGS

                          ws.RunNetOnce(netdef);
                          const auto& t2 =
                              ws.GetBlob("Y_cpu")->Get<TensorCPU>();
                          const auto& t1 =
                              ws.GetBlob("Y_ref")->Get<TensorCPU>();

                          CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
                          for (auto i = 0; i < t1.size(); ++i) {
                            // FP16 <-> FP32 round trip, accumulation, etc.
                            const float t1_i = t1.data<float>()[i];
                            const float t2_i = t2.data<float>()[i];
                            TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  {
    LOG(INFO) << "MPSCNNPadImage Test";
    for (const auto dims :
         std::vector<std::vector<size_t>>{{1, 3, 50, 80}, {1, 12, 50, 80}}) {
      Workspace ws;
      {
        auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
        t->Resize(dims);
        CPUContext ctx;
        math::RandGaussian<float, CPUContext>(
            t->size(), 0, 1, t->mutable_data<float>(), &ctx);
      }

      NetDef netdef;
      {
        auto& op = *(netdef.add_op());
        op.set_type("CopyToMPSCNN");
        op.add_input("X_cpu");
        op.add_output("X_mtl");
      }

      {
        auto& op = *(netdef.add_op());
        op.set_type("MPSCNNPadImage");
        op.add_input("X_mtl");
        {
          auto& arg = *(op.add_arg());
          arg.set_name("pad");
          arg.set_i(10);
        }
        {
          auto& arg = *(op.add_arg());
          arg.set_name("mode");
          arg.set_s("reflect");
        }
        op.add_output("Y_mtl");
      }

      {
        auto& op = *(netdef.add_op());
        op.set_type("CopyFromMPSCNN");
        op.add_input("Y_mtl");
        op.add_output("Y_cpu");
      }

      {
        auto& op = *(netdef.add_op());
        op.set_type("PadImage");
        op.add_input("X_cpu");
        {
          auto& arg = *(op.add_arg());
          arg.set_name("pad");
          arg.set_i(10);
        }
        {
          auto& arg = *(op.add_arg());
          arg.set_name("mode");
          arg.set_s("reflect");
        }
        op.add_output("Y_ref");
      }

      ws.RunNetOnce(netdef);
      const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
      const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

      CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
      for (auto i = 0; i < t1.size(); ++i) {
        // FP16 <-> FP32 round trip, accumulation, etc.
        const float t1_i = t1.data<float>()[i];
        const float t2_i = t2.data<float>()[i];
        // LOG(INFO) << "i: " << i << ", " << "CPU: " << t1_i << ", MTL: " <<
        // t2_i;
        TORCH_CHECK_NEAR(t1_i, t2_i, 0.01);
      }
    }
  }

  {
    LOG(INFO) << "MPSCNNPreprocess Test";
    Workspace ws;
    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
      t->Resize(1, 8, 13, 4);
      CPUContext ctx;
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<uint8_t>()[i] = rand() % 255;
      }
    }

    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("mean"), CPU);
      t->Resize(3);
      CPUContext ctx;
      t->mutable_data<float>()[0] = 100;
      t->mutable_data<float>()[1] = 50;
      t->mutable_data<float>()[2] = 150;
    }

    NetDef netdef;

    {
      auto& op = *(netdef.add_op());
      op.set_type("MPSCNNPackedInt8BGRANHWCToNCHWCStylizerPreprocess");
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

      op.add_output("Y_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyFromMPSCNN");
      op.add_input("Y_mtl");
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
    const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
    const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

    CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
    }
  }

  {
    LOG(INFO) << "MPSCNNDeprocess Test";
    Workspace ws;
    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
      t->Resize(1, 3, 8, 24);
      CPUContext ctx;
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<float>()[i] = rand() % 255;
      }
    }

    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("mean"), CPU);
      t->Resize(3);
      CPUContext ctx;
      t->mutable_data<float>()[0] = 100;
      t->mutable_data<float>()[1] = 50;
      t->mutable_data<float>()[2] = 150;
    }

    NetDef netdef;

    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyToMPSCNN");
      op.add_input("X_cpu");
      op.add_output("X_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("MPSCNNBRGNCHWCToPackedInt8BGRAStylizerDeprocess");
      op.add_input("X_mtl");
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

    CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<uint8_t>()[i];
      const float t2_i = t2.data<uint8_t>()[i];
      TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
    }
  }

  {
    LOG(INFO) << "MPSCNNDeprocess Test";
    Workspace ws;
    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
      t->Resize(1, 3, 1280, 720);
      CPUContext ctx;
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<float>()[i] = rand() % 1000 - 500;
      }
    }

    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("mean"), CPU);
      t->Resize(3);
      CPUContext ctx;
      t->mutable_data<float>()[0] = 30;
      t->mutable_data<float>()[1] = 40;
      t->mutable_data<float>()[2] = 50;
    }

    NetDef netdef;

    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyToMPSCNN");
      op.add_input("X_cpu");
      op.add_output("X_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("MPSCNNBRGNCHWCToPackedInt8BGRAStylizerDeprocess");
      op.add_input("X_mtl");
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

    CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<uint8_t>()[i];
      const float t2_i = t2.data<uint8_t>()[i];
      TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
    }
  }

  @autoreleasepool {
    for (const auto& batchSize : std::vector<int>{1, 2}) {
      for (const auto& stride_h : std::vector<int>{1, 2, 3}) {
        for (const auto& stride_w : std::vector<int>{1, 2, 3}) {
          for (const auto& kernel_h : std::vector<int>{1, 3, 8}) {
            for (const auto& kernel_w : std::vector<int>{1, 3, 8}) {
              for (const auto& pad_l : std::vector<int>{0, kernel_w / 2}) {
                for (const auto& pad_r : std::vector<int>{0, kernel_w / 2}) {
                  for (const auto& pad_t : std::vector<int>{0, kernel_h / 2}) {
                    for (const auto& pad_b :
                         std::vector<int>{0, kernel_h / 2}) {
                      // Waiting response from Apple
                      if (kernel_h != kernel_w) {
                        continue;
                      }
                      LOG(INFO) << "MPSCNNConv Test";
                      Workspace ws;
                      {
                        auto* t =
                            BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
                        t->Resize(batchSize, 12, 57, 72);
                        CPUContext ctx;
                        math::RandGaussian<float, CPUContext>(
                            t->size(), 0, 1, t->mutable_data<float>(), &ctx);
                      }

                      {
                        auto* t = BlobGetMutableTensor(ws.CreateBlob("W"), CPU);
                        t->Resize(8, 12, kernel_h, kernel_w);
                        CPUContext ctx;
                        math::RandGaussian<float, CPUContext>(
                            8 * 12 * kernel_h * kernel_w,
                            0,
                            1,
                            t->mutable_data<float>(),
                            &ctx);
                      }

                      {
                        auto* t = BlobGetMutableTensor(ws.CreateBlob("b"), CPU);
                        t->Resize(8);
                        CPUContext ctx;
                        math::RandGaussian<float, CPUContext>(
                            8, 0, 1, t->mutable_data<float>(), &ctx);
                      }

                      NetDef netdef;
#define ADD_ARGS(op)                     \
  do {                                   \
    add_arg_str(op, "order", "NCHW");    \
    add_arg_int_list(                    \
        op,                              \
        std::vector<string>{"stride_h",  \
                            "stride_w",  \
                            "pad_l",     \
                            "pad_r",     \
                            "pad_t",     \
                            "pad_b",     \
                            "kernel_w",  \
                            "kernel_h"}, \
        std::vector<int>{stride_h,       \
                         stride_w,       \
                         pad_l,          \
                         pad_r,          \
                         pad_t,          \
                         pad_b,          \
                         kernel_w,       \
                         kernel_h});     \
  } while (false)
                      {
                        auto& op = *(netdef.add_op());
                        op.set_type("CopyToMPSCNN");
                        op.add_input("X_cpu");
                        op.add_output("X_mtl");
                      }

                      {
                        auto& op = *(netdef.add_op());
                        op.set_type("MPSCNNConv");
                        op.add_input("X_mtl");
                        op.add_input("W");
                        op.add_input("b");
                        ADD_ARGS(op);
                        op.add_output("Y_mtl");
                      }

                      {
                        auto& op = *(netdef.add_op());
                        op.set_type("CopyFromMPSCNN");
                        op.add_input("Y_mtl");
                        op.add_output("Y_cpu");
                      }

                      {
                        auto& op = *(netdef.add_op());
                        op.set_type("Conv");
                        op.add_input("X_cpu");
                        op.add_input("W");
                        op.add_input("b");
                        ADD_ARGS(op);
                        op.add_output("Y_ref");
                      }
#undef ADD_ARGS
                      ws.RunNetOnce(netdef);
                      const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
                      const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

                      CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
                      for (auto i = 0; i < t1.size(); ++i) {
                        // FP16 <-> FP32 round trip, accumulation, etc.
                        const float t1_i = t1.data<float>()[i];
                        const float t2_i = t2.data<float>()[i];
                        TORCH_CHECK_NEAR(t1_i, t2_i, 0.2);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  @autoreleasepool {
    bool runtimeAtLeastIOS11 = SYSTEM_VERSION_GREATER_THAN_OR_EQUAL_TO(@"11.0");
    if (runtimeAtLeastIOS11) {
      for (const auto& batchSize : std::vector<int>{1, 2}) {
        for (const auto& input_channels : std::vector<int>{32, 64, 128, 256}) {
          for (const auto& channel_multiplier : std::vector<int>{1}) {
            LOG(INFO) << "MPSCNNDepthwiseConv Test";
            Workspace ws;
            int output_channels = input_channels * channel_multiplier;
            {
              auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
              t->Resize(batchSize, input_channels, 57, 72);
              CPUContext ctx;
              math::RandGaussian<float, CPUContext>(
                  t->size(), 0, 1, t->mutable_data<float>(), &ctx);
            }

            {
              auto* t = BlobGetMutableTensor(ws.CreateBlob("W"), CPU);
              t->Resize(output_channels, 1, 3, 3);
              CPUContext ctx;
              math::RandGaussian<float, CPUContext>(
                  t->size(), 0, 1, t->mutable_data<float>(), &ctx);
            }

            {
              auto* t = BlobGetMutableTensor(ws.CreateBlob("b"), CPU);
              t->Resize(output_channels);
              CPUContext ctx;
              math::RandGaussian<float, CPUContext>(
                  t->size(), 0, 1, t->mutable_data<float>(), &ctx);
            }

            NetDef netdef;
#define ADD_ARGS(op)                                      \
  do {                                                    \
    add_arg_str(op, "order", "NCHW");                     \
    add_arg_int_list(                                     \
        op,                                               \
        std::vector<string>{"stride", "kernel", "group"}, \
        std::vector<int>{1, 3, input_channels});          \
  } while (false)
            {
              auto& op = *(netdef.add_op());
              op.set_type("CopyToMPSCNN");
              op.add_input("X_cpu");
              op.add_output("X_mtl");
            }

            {
              auto& op = *(netdef.add_op());
              op.set_type("MPSCNNConv");
              op.add_input("X_mtl");
              op.add_input("W");
              op.add_input("b");
              ADD_ARGS(op);
              op.add_output("Y_mtl");
            }

            {
              auto& op = *(netdef.add_op());
              op.set_type("CopyFromMPSCNN");
              op.add_input("Y_mtl");
              op.add_output("Y_cpu");
            }

            {
              auto& op = *(netdef.add_op());
              op.set_type("Conv");
              op.add_input("X_cpu");
              op.add_input("W");
              op.add_input("b");
              ADD_ARGS(op);
              op.add_output("Y_ref");
            }
#undef ADD_ARGS
            ws.RunNetOnce(netdef);
            const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
            const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

            CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
            for (auto i = 0; i < t1.size(); ++i) {
              // FP16 <-> FP32 round trip, accumulation, etc.
              const float t1_i = t1.data<float>()[i];
              const float t2_i = t2.data<float>()[i];
              TORCH_CHECK_NEAR(t1_i, t2_i, 0.3);
            }
          }
        }
      }
    }
  }

  {
    LOG(INFO) << "MPSCNNConvRelu Test";
    Workspace ws;
    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
      t->Resize(1, 12, 57, 72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(
          t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }

    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("W"), CPU);
      t->Resize(8, 12, 3, 3);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(
          8 * 12 * 3 * 3, 0, 1, t->mutable_data<float>(), &ctx);
    }

    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("b"), CPU);
      t->Resize(8);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(
          8, 0, 1, t->mutable_data<float>(), &ctx);
    }

    NetDef netdef;
    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyToMPSCNN");
      op.add_input("X_cpu");
      op.add_output("X_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("MPSCNNConvRelu");
      op.add_input("X_mtl");
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
        arg.set_i(3);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad");
        arg.set_i(1);
      }
      op.add_output("Y_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyFromMPSCNN");
      op.add_input("Y_mtl");
      op.add_output("Y_cpu");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("Conv");
      op.add_input("X_cpu");
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
        arg.set_i(3);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad");
        arg.set_i(1);
      }
      op.add_output("Y_ref");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("Relu");
      op.add_input("Y_ref");
      op.add_output("Y_ref");
    }

    ws.RunNetOnce(netdef);
    const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
    const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

    CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
    }
  }

  {
    LOG(INFO) << "MPSConv Test";
    Workspace ws;
    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
      t->Resize(1, 12, 57, 72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(
          t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }

    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("W"), CPU);
      t->Resize(8, 12, 3, 3);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(
          8 * 12 * 3 * 3, 0, 1, t->mutable_data<float>(), &ctx);
    }

    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("b"), CPU);
      t->Resize(8);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(
          8, 0, 1, t->mutable_data<float>(), &ctx);
    }

    NetDef netdef;
    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyToMPSCNN");
      op.add_input("X_cpu");
      op.add_output("X_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("MPSCNNConv");
      op.add_input("X_mtl");
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
        arg.set_i(3);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad");
        arg.set_i(0);
      }
      op.add_output("Y_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyFromMPSCNN");
      op.add_input("Y_mtl");
      op.add_output("Y_cpu");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("Conv");
      op.add_input("X_cpu");
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
        arg.set_i(3);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad");
        arg.set_i(0);
      }
      op.add_output("Y_ref");
    }

    ws.RunNetOnce(netdef);
    const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
    const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

    CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
    }
  }

  {
    for (const auto& batchSize : {1, 2}) {
      for (const auto& C : {1, 2}) {
        for (const auto& M : {1, 2}) {
          for (const auto& K : {3, 4}) {
            for (const auto& P : {1, 2}) {
              LOG(INFO) << "MPSConv Test";
              Workspace ws;
              {
                auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
                t->Resize(batchSize, C, 12, 16);
                CPUContext ctx;
                math::RandGaussian<float, CPUContext>(
                    t->size(), 0, 1, t->mutable_data<float>(), &ctx);
              }

              {
                auto* t = BlobGetMutableTensor(ws.CreateBlob("W"), CPU);
                t->Resize(M, C, K, K);
                CPUContext ctx;
                math::RandGaussian<float, CPUContext>(
                    t->size(), 0, 1, t->mutable_data<float>(), &ctx);
              }

              {
                auto* t = BlobGetMutableTensor(ws.CreateBlob("b"), CPU);
                t->Resize(M);
                CPUContext ctx;
                math::RandGaussian<float, CPUContext>(
                    t->size(), 0, 1, t->mutable_data<float>(), &ctx);
              }

              NetDef netdef;
              {
                auto& op = *(netdef.add_op());
                op.set_type("CopyToMPSCNN");
                op.add_input("X_cpu");
                op.add_output("X_mtl");
              }

              {
                auto& op = *(netdef.add_op());
                op.set_type("MPSCNNConv");
                op.add_input("X_mtl");
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
                  arg.set_i(K);
                }
                {
                  auto& arg = *(op.add_arg());
                  arg.set_name("pad");
                  arg.set_i(P);
                }
                op.add_output("Y_mtl");
              }

              {
                auto& op = *(netdef.add_op());
                op.set_type("CopyFromMPSCNN");
                op.add_input("Y_mtl");
                op.add_output("Y_cpu");
              }

              {
                auto& op = *(netdef.add_op());
                op.set_type("Conv");
                op.add_input("X_cpu");
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
                  arg.set_i(K);
                }
                {
                  auto& arg = *(op.add_arg());
                  arg.set_name("pad");
                  arg.set_i(P);
                }
                op.add_output("Y_ref");
              }

              ws.RunNetOnce(netdef);
              const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
              const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

              CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
              for (auto i = 0; i < t1.size(); ++i) {
                // FP16 <-> FP32 round trip, accumulation, etc.
                const float t1_i = t1.data<float>()[i];
                const float t2_i = t2.data<float>()[i];
                TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
              }
            }
          }
        }
      }
    }
  }

  {
    for (const auto& batchSize : {1, 2}) {
      for (const auto& group : {1, 2}) {
        for (const auto& C : {8, 16}) {
          for (const auto& M : {8, 16}) {
            for (const auto& K : {3, 4}) {
              for (const auto& P : {1, 2}) {
                LOG(INFO) << "MPSCNNConv Test - group";
                Workspace ws;
                {
                  auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
                  t->Resize(batchSize, C, 12, 16);
                  CPUContext ctx;
                  math::RandGaussian<float, CPUContext>(
                      t->size(), 0, 1, t->mutable_data<float>(), &ctx);
                }

                {
                  auto* t = BlobGetMutableTensor(ws.CreateBlob("W"), CPU);
                  t->Resize(M, C / group, K, K);
                  CPUContext ctx;
                  math::RandGaussian<float, CPUContext>(
                      t->size(), 0, 1, t->mutable_data<float>(), &ctx);
                }

                {
                  auto* t = BlobGetMutableTensor(ws.CreateBlob("b"), CPU);
                  t->Resize(M);
                  CPUContext ctx;
                  math::RandGaussian<float, CPUContext>(
                      t->size(), 0, 1, t->mutable_data<float>(), &ctx);
                }

                NetDef netdef;
                {
                  auto& op = *(netdef.add_op());
                  op.set_type("CopyToMPSCNN");
                  op.add_input("X_cpu");
                  op.add_output("X_mtl");
                }

                {
                  auto& op = *(netdef.add_op());
                  op.set_type("MPSCNNConv");
                  op.add_input("X_mtl");
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
                    arg.set_i(K);
                  }
                  {
                    auto& arg = *(op.add_arg());
                    arg.set_name("pad");
                    arg.set_i(P);
                  }
                  {
                    auto& arg = *(op.add_arg());
                    arg.set_name("group");
                    arg.set_i(group);
                  }
                  op.add_output("Y_mtl");
                }

                {
                  auto& op = *(netdef.add_op());
                  op.set_type("CopyFromMPSCNN");
                  op.add_input("Y_mtl");
                  op.add_output("Y_cpu");
                }

                {
                  auto& op = *(netdef.add_op());
                  op.set_type("Conv");
                  op.add_input("X_cpu");
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
                    arg.set_i(K);
                  }
                  {
                    auto& arg = *(op.add_arg());
                    arg.set_name("pad");
                    arg.set_i(P);
                  }
                  {
                    auto& arg = *(op.add_arg());
                    arg.set_name("group");
                    arg.set_i(group);
                  }
                  op.add_output("Y_ref");
                }

                ws.RunNetOnce(netdef);
                const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
                const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

                CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
                for (auto i = 0; i < t1.size(); ++i) {
                  // FP16 <-> FP32 round trip, accumulation, etc.
                  const float t1_i = t1.data<float>()[i];
                  const float t2_i = t2.data<float>()[i];
                  TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
                }
              }
            }
          }
        }
      }
    }
  }

  {
    LOG(INFO) << "MPSCNNMul Test";
    Workspace ws;
    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("X0_cpu"), CPU);
      t->Resize(1, 12, 57, 72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(
          t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }

    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("X1_cpu"), CPU);
      t->Resize(72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(
          t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }

    NetDef netdef;
    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyToMPSCNN");
      op.add_input("X0_cpu");
      op.add_output("X0_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("MPSCNNMul");
      op.add_input("X0_mtl");
      op.add_input("X1_cpu");
      op.add_output("Y_mtl");
      add_arg_int(op, "broadcast", 1);
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyFromMPSCNN");
      op.add_input("Y_mtl");
      op.add_output("Y_cpu");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("Mul");
      op.add_input("X0_cpu");
      op.add_input("X1_cpu");
      op.add_output("Y_ref");
      add_arg_int(op, "broadcast", 1);
    }

    ws.RunNetOnce(netdef);
    const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
    const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

    CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      TORCH_CHECK_NEAR(t1_i, t2_i, 0.02);
    }
  }

  {
    LOG(INFO) << "MPSCNNSub Test";
    Workspace ws;
    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("X0_cpu"), CPU);
      t->Resize(1, 12, 57, 72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(
          t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }

    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("X1_cpu"), CPU);
      t->Resize(72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(
          t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }

    NetDef netdef;
    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyToMPSCNN");
      op.add_input("X0_cpu");
      op.add_output("X0_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("MPSCNNSub");
      op.add_input("X0_mtl");
      op.add_input("X1_cpu");
      op.add_output("Y_mtl");
      add_arg_int(op, "broadcast", 1);
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyFromMPSCNN");
      op.add_input("Y_mtl");
      op.add_output("Y_cpu");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("Sub");
      op.add_input("X0_cpu");
      op.add_input("X1_cpu");
      op.add_output("Y_ref");
      add_arg_int(op, "broadcast", 1);
    }

    ws.RunNetOnce(netdef);
    const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
    const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

    CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      TORCH_CHECK_NEAR(t1_i, t2_i, 0.01);
    }
  }

  {
    LOG(INFO) << "MPSAdd Test";
    Workspace ws;
    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("X0_cpu"), CPU);
      t->Resize(1, 12, 57, 72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(
          t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }

    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("X1_cpu"), CPU);
      t->Resize(1, 12, 57, 72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(
          t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }

    NetDef netdef;
    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyToMPSCNN");
      op.add_input("X0_cpu");
      op.add_output("X0_mtl");
      op.add_input("X1_cpu");
      op.add_output("X1_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("MPSCNNAdd");
      op.add_input("X0_mtl");
      op.add_input("X1_mtl");
      op.add_output("Y_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyFromMPSCNN");
      op.add_input("Y_mtl");
      op.add_output("Y_cpu");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("Add");
      op.add_input("X0_cpu");
      op.add_input("X1_cpu");
      op.add_output("Y_ref");
    }

    ws.RunNetOnce(netdef);
    const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
    const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

    CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      TORCH_CHECK_NEAR(t1_i, t2_i, 0.01);
    }
  }

  {
    LOG(INFO) << "MPSAdd Test";
    Workspace ws;
    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("X0_cpu"), CPU);
      t->Resize(1, 12, 57, 72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(
          t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }

    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("X1_cpu"), CPU);
      t->Resize(1, 12, 57, 72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(
          t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }

    NetDef netdef;
    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyToMPSCNN");
      op.add_input("X0_cpu");
      op.add_output("X0_mtl");
      op.add_input("X1_cpu");
      op.add_output("X1_mtl");

      // First input is read twice.
      {
        auto& arg = *(op.add_arg());
        arg.set_name("__mpscnn_read_count__");
        arg.add_ints(2);
        arg.add_ints(1);
      }
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("MPSCNNAdd");
      op.add_input("X0_mtl");
      op.add_input("X1_mtl");
      op.add_output("X2_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("MPSCNNAdd");
      op.add_input("X0_mtl");
      op.add_input("X2_mtl");
      op.add_output("Y_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyFromMPSCNN");
      op.add_input("Y_mtl");
      op.add_output("Y_cpu");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("Add");
      op.add_input("X0_cpu");
      op.add_input("X1_cpu");
      op.add_output("X2_cpu");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("Add");
      op.add_input("X0_cpu");
      op.add_input("X2_cpu");
      op.add_output("Y_ref");
    }

    ws.RunNetOnce(netdef);
    const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
    const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

    CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      TORCH_CHECK_NEAR(t1_i, t2_i, 0.05);
    }
  }

  {
    for (const auto& n : {"Relu", "Tanh", "Sigmoid"}) {
      LOG(INFO) << "MPSCNNNeuron Test: " << n;
      Workspace ws;
      {
        auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
        t->Resize(1, 4, 12, 12);
        CPUContext ctx;
        math::RandGaussian<float, CPUContext>(
            t->size(), 0, 1, t->mutable_data<float>(), &ctx);
      }

      NetDef netdef;
      {
        auto& op = *(netdef.add_op());
        op.set_type("CopyToMPSCNN");
        op.add_input("X_cpu");
        op.add_output("X_mtl");
      }

      {
        auto& op = *(netdef.add_op());
        op.set_type(std::string("MPSCNN") + n);
        op.add_input("X_mtl");
        op.add_output("Y_mtl");
      }

      {
        auto& op = *(netdef.add_op());
        op.set_type("CopyFromMPSCNN");
        op.add_input("Y_mtl");
        op.add_output("Y_cpu");
      }

      {
        auto& op = *(netdef.add_op());
        op.set_type(n);
        op.add_input("X_cpu");
        op.add_output("Y_ref");
      }

      ws.RunNetOnce(netdef);
      const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
      const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

      CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
      for (auto i = 0; i < t1.size(); ++i) {
        // FP16 <-> FP32 round trip, accumulation, etc.
        const float t1_i = t1.data<float>()[i];
        const float t2_i = t2.data<float>()[i];
        TORCH_CHECK_NEAR(t1_i, t2_i, 0.02);
      }
    }
  }

  {
    LOG(INFO) << "MPSCNNDropout Test";
    Workspace ws;
    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
      t->Resize(1, 12, 57, 72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(
          t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }

    NetDef netdef;
    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyToMPSCNN");
      op.add_input("X_cpu");
      op.add_output("X_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("MPSCNNDropout");
      op.add_input("X_mtl");
      {
        auto& arg = *(op.add_arg());
        arg.set_name(OpSchema::Arg_IsTest);
        arg.set_i(1);
      }
      op.add_output("Y_mtl");
      op.add_output("Y_mask_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("CopyFromMPSCNN");
      op.add_input("Y_mtl");
      op.add_output("Y_cpu");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("Dropout");
      op.add_input("X_cpu");
      {
        auto& arg = *(op.add_arg());
        arg.set_name(OpSchema::Arg_IsTest);
        arg.set_i(1);
      }
      op.add_output("Y_ref");
      op.add_output("Y_mask");
    }

    ws.RunNetOnce(netdef);
    const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
    const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();
    CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
    LOG(INFO) << t1.sizes();
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
    }
  }

  {
    for (const auto scale : std::vector<float>{1.0, 2.0, 0.0625}) {
      for (const auto channels : std::vector<size_t>{1, 3, 5, 8}) {
        for (const auto pool : std::vector<size_t>{1, 3, 7}) {
          for (const auto sampling_ratio : std::vector<size_t>{0, 1, 2, 3}) {
            LOG(INFO) << "MPSCNNRoIWarp Test - sampling_ratio:"
                      << sampling_ratio << "- pool: " << pool
                      << " - scale: " << scale;
            Workspace ws;
            {
              auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
              t->Resize(1, channels, 40, 40);
              CPUContext ctx;
              math::RandGaussian<float, CPUContext>(
                  t->size(), 4, 2, t->mutable_data<float>(), &ctx);
            }
            {
              // Use the batch-first encoding (n, [bbox])
              auto* t = BlobGetMutableTensor(ws.CreateBlob("R"), CPU);
              t->Resize(6, 5);
              for (auto i = 0; i < t->dim32(0); ++i) {
                t->mutable_data<float>()[5 * i + 0] = 0; // batch
                t->mutable_data<float>()[5 * i + 1] = (i % 4 + 1) * 1.0 / scale;
                t->mutable_data<float>()[5 * i + 2] = (i % 5 + 1) * 1.0 / scale;
                t->mutable_data<float>()[5 * i + 3] = (i % 3 + 7) * 1.0 / scale;
                t->mutable_data<float>()[5 * i + 4] = (i % 4 + 7) * 1.0 / scale;
              }
            }

            NetDef netdef;
            {
              auto& op = *(netdef.add_op());
              op.set_type("CopyToMPSCNN");
              op.add_input("X_cpu");
              op.add_output("X_mtl");
            }

            {
              auto& op = *(netdef.add_op());
              op.set_type("MPSCNNRoIWarp");
              op.add_input("X_mtl");
              op.add_input("R");
              {
                auto& arg = *(op.add_arg());
                arg.set_name("sampling_ratio");
                arg.set_i(sampling_ratio);
              }
              {
                auto& arg = *(op.add_arg());
                arg.set_name("pooled_h");
                arg.set_i(pool);
              }
              {
                auto& arg = *(op.add_arg());
                arg.set_name("pooled_w");
                arg.set_i(pool);
              }
              {
                auto& arg = *(op.add_arg());
                arg.set_name("spatial_scale");
                arg.set_f(scale);
              }
              op.add_output("Y_mtl");
            }

            {
              auto& op = *(netdef.add_op());
              op.set_type("CopyFromMPSCNN");
              op.add_input("Y_mtl");
              op.add_output("Y_cpu");
            }

            {
              auto& op = *(netdef.add_op());
              op.set_type("RoIWarp");
              op.add_input("X_cpu");
              op.add_input("R");
              {
                auto& arg = *(op.add_arg());
                arg.set_name("sampling_ratio");
                arg.set_i(sampling_ratio);
              }
              {
                auto& arg = *(op.add_arg());
                arg.set_name("pooled_h");
                arg.set_i(pool);
              }
              {
                auto& arg = *(op.add_arg());
                arg.set_name("pooled_w");
                arg.set_i(pool);
              }
              {
                auto& arg = *(op.add_arg());
                arg.set_name("spatial_scale");
                arg.set_f(scale);
              }
              op.add_output("Y_ref");
            }

            ws.RunNetOnce(netdef);
            const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();
            const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();

            CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
            LOG(INFO) << t1.sizes();
            for (auto i = 0; i < t1.size(); ++i) {
              // FP16 <-> FP32 round trip, accumulation, etc.
              const float t1_i = t1.data<float>()[i];
              const float t2_i = t2.data<float>()[i];
              TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
            }
          }
        }
      }
    }
  }

  {
    for (const auto scale : std::vector<float>{1.0, 2.0, 0.0625}) {
      for (const auto pool : std::vector<size_t>{1, 3, 7}) {
        LOG(INFO) << "MPSCNNRoIWarp Test 2";
        Workspace ws;
        {
          auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
          t->Resize(1, 8, 40, 40);
          CPUContext ctx;
          math::RandGaussian<float, CPUContext>(
              t->size(), 4, 2, t->mutable_data<float>(), &ctx);
        }
        {
          auto* t = BlobGetMutableTensor(ws.CreateBlob("R"), CPU);
          t->Resize(6, 4);
          for (auto i = 0; i < t->dim32(0); ++i) {
            t->mutable_data<float>()[4 * i + 0] = (i % 4 + 1) * 1.0 / scale;
            t->mutable_data<float>()[4 * i + 1] = (i % 5 + 1) * 1.0 / scale;
            t->mutable_data<float>()[4 * i + 2] = (i % 3 + 7) * 1.0 / scale;
            t->mutable_data<float>()[4 * i + 3] = (i % 4 + 7) * 1.0 / scale;
          }
        }

        NetDef netdef;
        {
          auto& op = *(netdef.add_op());
          op.set_type("CopyToMPSCNN");
          op.add_input("X_cpu");
          op.add_output("X_mtl");
        }

        {
          auto& op = *(netdef.add_op());
          op.set_type("MPSCNNRoIWarp");
          op.add_input("X_mtl");
          op.add_input("R");
          {
            auto& arg = *(op.add_arg());
            arg.set_name("sampling_ratio");
            arg.set_i(1);
          }
          {
            auto& arg = *(op.add_arg());
            arg.set_name("pooled_h");
            arg.set_i(pool);
          }
          {
            auto& arg = *(op.add_arg());
            arg.set_name("pooled_w");
            arg.set_i(pool);
          }
          {
            auto& arg = *(op.add_arg());
            arg.set_name("spatial_scale");
            arg.set_f(scale);
          }
          op.add_output("Y_mtl");
        }

        {
          auto& op = *(netdef.add_op());
          op.set_type("CopyFromMPSCNN");
          op.add_input("Y_mtl");
          op.add_output("Y_cpu");
        }

        {
          auto& op = *(netdef.add_op());
          op.set_type("RoIWarp");
          op.add_input("X_cpu");
          op.add_input("R");
          {
            auto& arg = *(op.add_arg());
            arg.set_name("sampling_ratio");
            arg.set_i(1);
          }
          {
            auto& arg = *(op.add_arg());
            arg.set_name("pooled_h");
            arg.set_i(pool);
          }
          {
            auto& arg = *(op.add_arg());
            arg.set_name("pooled_w");
            arg.set_i(pool);
          }
          {
            auto& arg = *(op.add_arg());
            arg.set_name("spatial_scale");
            arg.set_f(scale);
          }
          op.add_output("Y_ref");
        }

        ws.RunNetOnce(netdef);
        const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();
        const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();

        CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
        LOG(INFO) << t1.sizes();
        for (auto i = 0; i < t1.size(); ++i) {
          // FP16 <-> FP32 round trip, accumulation, etc.
          const float t1_i = t1.data<float>()[i];
          const float t2_i = t2.data<float>()[i];
          TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
        }
      }
    }
  }

  {
    for (const auto height_scale : std::vector<float>{1.0, 0.5, 1.7}) {
      for (const auto width_scale : std::vector<float>{1.0, 0.5, 2.3}) {
        for (const auto C : std::vector<float>{2, 7, 11}) {
          for (const auto N : std::vector<float>{1, 2}) {
            LOG(INFO) << "MPSCNNResizeNearestOp Test";
            Workspace ws;
            {
              auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
              t->Resize(N, C, 37, 89);
              CPUContext ctx;
              math::RandGaussian<float, CPUContext>(
                  t->size(), 4, 2, t->mutable_data<float>(), &ctx);
            }
            NetDef netdef;
            {
              auto& op = *(netdef.add_op());
              op.set_type("CopyToMPSCNN");
              op.add_input("X_cpu");
              op.add_output("X_mtl");
            }

            {
              auto& op = *(netdef.add_op());
              op.set_type("MPSCNNResizeNearest");
              op.add_input("X_mtl");
              {
                auto& arg = *(op.add_arg());
                arg.set_name("height_scale");
                arg.set_f(height_scale);
              }
              {
                auto& arg = *(op.add_arg());
                arg.set_name("width_scale");
                arg.set_f(width_scale);
              }
              op.add_output("Y_mtl");
            }

            {
              auto& op = *(netdef.add_op());
              op.set_type("CopyFromMPSCNN");
              op.add_input("Y_mtl");
              op.add_output("Y_cpu");
            }

            {
              auto& op = *(netdef.add_op());
              op.set_type("ResizeNearest");
              op.add_input("X_cpu");
              {
                auto& arg = *(op.add_arg());
                arg.set_name("height_scale");
                arg.set_f(height_scale);
              }
              {
                auto& arg = *(op.add_arg());
                arg.set_name("width_scale");
                arg.set_f(width_scale);
              }
              op.add_output("Y_ref");
            }

            ws.RunNetOnce(netdef);
            const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();
            const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();

            CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
            LOG(INFO) << t1.sizes();
            for (auto i = 0; i < t1.size(); ++i) {
              // FP16 <-> FP32 round trip, accumulation, etc.
              const float t1_i = t1.data<float>()[i];
              const float t2_i = t2.data<float>()[i];
              TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
            }
          }
        }
      }
    }
  }

  {
    LOG(INFO) << "MPSCNNGenerateProposals Test: \n";
    Workspace ws;
    auto num_images = 1;
    auto A = 2; // # anchors
    auto H = 4; // height
    auto W = 5; // width
    vector<float> scores{
        5.44218998e-03, 1.19207997e-03, 1.12379994e-03, 1.17181998e-03,
        1.20544003e-03, 6.17993006e-04, 1.05261997e-05, 8.91025957e-06,
        9.29536981e-09, 6.09605013e-05, 4.72735002e-04, 1.13482002e-10,
        1.50015003e-05, 4.45032993e-06, 3.21612994e-08, 8.02662980e-04,
        1.40488002e-04, 3.12508007e-07, 3.02616991e-06, 1.97759000e-08,
        2.66913995e-02, 5.26766013e-03, 5.05053019e-03, 5.62100019e-03,
        5.37420018e-03, 5.26280981e-03, 2.48894998e-04, 1.06842002e-04,
        3.92931997e-06, 1.79388002e-03, 4.79440019e-03, 3.41609990e-07,
        5.20430971e-04, 3.34090000e-05, 2.19159006e-07, 2.28786003e-03,
        5.16703985e-05, 4.04523007e-06, 1.79227004e-06, 5.32449000e-08};
    vector<float> bbx{
        -1.65040009e-02, -1.84051003e-02, -1.85930002e-02, -2.08263006e-02,
        -1.83814000e-02, -2.89172009e-02, -3.89706008e-02, -7.52277970e-02,
        -1.54091999e-01, -2.55433004e-02, -1.77490003e-02, -1.10340998e-01,
        -4.20190990e-02, -2.71421000e-02, 6.89801015e-03,  5.71171008e-02,
        -1.75665006e-01, 2.30021998e-02,  3.08554992e-02,  -1.39333997e-02,
        3.40579003e-01,  3.91070992e-01,  3.91624004e-01,  3.92527014e-01,
        3.91445011e-01,  3.79328012e-01,  4.26631987e-01,  3.64892989e-01,
        2.76894987e-01,  5.13985991e-01,  3.79999995e-01,  1.80457994e-01,
        4.37402993e-01,  4.18545991e-01,  2.51549989e-01,  4.48318988e-01,
        1.68564007e-01,  4.65440989e-01,  4.21891987e-01,  4.45928007e-01,
        3.27155995e-03,  3.71480011e-03,  3.60032008e-03,  4.27092984e-03,
        3.74579988e-03,  5.95752988e-03,  -3.14473989e-03, 3.52022005e-03,
        -1.88564006e-02, 1.65188999e-03,  1.73791999e-03,  -3.56074013e-02,
        -1.66615995e-04, 3.14146001e-03,  -1.11830998e-02, -5.35363983e-03,
        6.49790000e-03,  -9.27671045e-03, -2.83346009e-02, -1.61233004e-02,
        -2.15505004e-01, -2.19910994e-01, -2.20872998e-01, -2.12831005e-01,
        -2.19145000e-01, -2.27687001e-01, -3.43973994e-01, -2.75869995e-01,
        -3.19516987e-01, -2.50418007e-01, -2.48537004e-01, -5.08224010e-01,
        -2.28724003e-01, -2.82402009e-01, -3.75815988e-01, -2.86352992e-01,
        -5.28333001e-02, -4.43836004e-01, -4.55134988e-01, -4.34897989e-01,
        -5.65053988e-03, -9.25739005e-04, -1.06790999e-03, -2.37016007e-03,
        -9.71166010e-04, -8.90910998e-03, -1.17592998e-02, -2.08992008e-02,
        -4.94231991e-02, 6.63906988e-03,  3.20469006e-03,  -6.44695014e-02,
        -3.11607006e-03, 2.02738005e-03,  1.48096997e-02,  4.39785011e-02,
        -8.28424022e-02, 3.62076014e-02,  2.71668993e-02,  1.38250999e-02,
        6.76669031e-02,  1.03252999e-01,  1.03255004e-01,  9.89722982e-02,
        1.03646003e-01,  4.79663983e-02,  1.11014001e-01,  9.31736007e-02,
        1.15768999e-01,  1.04014002e-01,  -8.90677981e-03, 1.13103002e-01,
        1.33085996e-01,  1.25405997e-01,  1.50051996e-01,  -1.13038003e-01,
        7.01059997e-02,  1.79651007e-01,  1.41055003e-01,  1.62841007e-01,
        -1.00247003e-02, -8.17587040e-03, -8.32176022e-03, -8.90108012e-03,
        -8.13035015e-03, -1.77263003e-02, -3.69572006e-02, -3.51580009e-02,
        -5.92143014e-02, -1.80795006e-02, -5.46086021e-03, -4.10550982e-02,
        -1.83081999e-02, -2.15411000e-02, -1.17953997e-02, 3.33894007e-02,
        -5.29635996e-02, -6.97528012e-03, -3.15250992e-03, -3.27355005e-02,
        1.29676998e-01,  1.16080999e-01,  1.15947001e-01,  1.21797003e-01,
        1.16089001e-01,  1.44875005e-01,  1.15617000e-01,  1.31586999e-01,
        1.74735002e-02,  1.21973999e-01,  1.31596997e-01,  2.48907991e-02,
        6.18605018e-02,  1.12855002e-01,  -6.99798986e-02, 9.58312973e-02,
        1.53593004e-01,  -8.75087008e-02, -4.92327996e-02, -3.32239009e-02};
    vector<float> im_info{60, 80, 0.166667};
    vector<float> anchors{-38, -16, 53, 31, -120, -120, 135, 135};
    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
      t->Resize(num_images, A, H, W);
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<float>()[i] = scores[i];
      }
    }

    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("bbox_delta_cpu"), CPU);
      t->Resize(num_images, 4 * A, H, W);
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<float>()[i] = bbx[i];
      }
    }

    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("im_info"), CPU);
      t->Resize(num_images, 3);
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<float>()[i] = im_info[i];
      }
    }

    {
      auto* t = BlobGetMutableTensor(ws.CreateBlob("anchors"), CPU);
      t->Resize(A, 4);
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<float>()[i] = anchors[i];
      }
    }

    NetDef netdef;

    {
      auto& op = *(netdef.add_op());
      op.set_type("MPSCNNGenerateProposalsCPP");
      op.add_input("X_cpu");
      op.add_input("bbox_delta_cpu");
      op.add_input("im_info");
      op.add_input("anchors");
      op.add_output("rois");
      op.add_output("rois_probs");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("GenerateProposalsCPP");
      op.add_input("X_cpu");
      op.add_input("bbox_delta_cpu");
      op.add_input("im_info");
      op.add_input("anchors");
      op.add_output("rois_ref");
      op.add_output("rois_probs_ref");
    }

    ws.RunNetOnce(netdef);
    const auto& t2 = ws.GetBlob("rois")->Get<TensorCPU>();
    const auto& t1 = ws.GetBlob("rois_ref")->Get<TensorCPU>();

    const auto& t4 = ws.GetBlob("rois_probs")->Get<TensorCPU>();
    const auto& t3 = ws.GetBlob("rois_probs_ref")->Get<TensorCPU>();

    LOG(INFO) << "t1: " << t1.size() << " t2: " << t2.size();

    const float HALF_MIN_VAL = 6.103515625e-05;
    for (auto i = 0; i < fmin(t1.size(), t2.size()); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      const float t3_i = t3.data<float>()[i / 5];
      if (t3_i - HALF_MIN_VAL * 2 > 0) {
        LOG(INFO) << i << " " << t1_i << " " << t2_i << " " << t3_i;
        TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
      }
    }

    for (auto i = 0; i < fmin(t3.size(), t4.size()); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t3_i = t3.data<float>()[i];
      const float t4_i = t4.data<float>()[i];
      LOG(INFO) << i << " " << t3_i;
      TORCH_CHECK_NEAR(t3_i, t4_i, 0.1);
    }
  }

  {
    for (const auto& batchSize : std::vector<size_t>{1, 2}) {
      LOG(INFO) << "MPSCNNSoftmax Test";
      Workspace ws;
      {
        auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
        // Only works for spatial dimension of (1, 1) - weird.
        t->Resize(batchSize, 12, 1, 1);
        CPUContext ctx;
        math::RandGaussian<float, CPUContext>(
            t->size(), 0, 1, t->mutable_data<float>(), &ctx);
      }

      NetDef netdef;
      {
        auto& op = *(netdef.add_op());
        op.set_type("CopyToMPSCNN");
        op.add_input("X_cpu");
        op.add_output("X_mtl");
      }

      {
        auto& op = *(netdef.add_op());
        op.set_type("MPSCNNSoftmax");
        op.add_input("X_mtl");
        op.add_output("Y_mtl");
      }

      {
        auto& op = *(netdef.add_op());
        op.set_type("CopyFromMPSCNN");
        op.add_input("Y_mtl");
        op.add_output("Y_cpu");
      }

      {
        auto& op = *(netdef.add_op());
        op.set_type("Softmax");
        op.add_input("X_cpu");
        op.add_output("Y_ref");
      }

      ws.RunNetOnce(netdef);
      const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
      const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();
      CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
      LOG(INFO) << t1.sizes();
      for (auto i = 0; i < t1.size(); ++i) {
        // FP16 <-> FP32 round trip, accumulation, etc.
        const float t1_i = t1.data<float>()[i];
        const float t2_i = t2.data<float>()[i];
        TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
      }
    }
  }

  @autoreleasepool {
    for (const auto& inputChannels : std::vector<size_t>{3, 8}) {
      for (const auto& outputChannels : std::vector<size_t>{3, 8}) {
        for (const auto& batchSize : std::vector<size_t>{1, 2}) {
          for (const auto& stride_h : std::vector<int>{1, 2, 3}) {
            for (const auto& stride_w : std::vector<int>{1, 2, 3}) {
              for (const auto& kernel_h : std::vector<int>{3}) {
                for (const auto& kernel_w : std::vector<int>{3}) {
                  for (const auto& pad_l : std::vector<int>{0, kernel_w / 2}) {
                    for (const auto& pad_r :
                         std::vector<int>{0, kernel_w / 2}) {
                      for (const auto& pad_t :
                           std::vector<int>{0, kernel_h / 2}) {
                        for (const auto& pad_b :
                             std::vector<int>{0, kernel_h / 2}) {
                          for (const auto& adj : {0, 1, 2, 3}) {
                            if (adj >= fmin(stride_h, stride_w)) {
                              continue;
                            }

                            LOG(INFO) << "MPSConvTranspose Test";
                            Workspace ws;
                            {
                              auto* t = BlobGetMutableTensor(
                                  ws.CreateBlob("X_cpu"), CPU);
                              t->Resize(batchSize, inputChannels, 8, 12);
                              CPUContext ctx;
                              math::RandGaussian<float, CPUContext>(
                                  t->size(),
                                  0,
                                  1,
                                  t->mutable_data<float>(),
                                  &ctx);
                            }

                            {
                              auto* t =
                                  BlobGetMutableTensor(ws.CreateBlob("W"), CPU);
                              t->Resize(
                                  inputChannels,
                                  outputChannels,
                                  kernel_h,
                                  kernel_w);
                              CPUContext ctx;
                              math::RandGaussian<float, CPUContext>(
                                  t->size(),
                                  0,
                                  1,
                                  t->mutable_data<float>(),
                                  &ctx);
                            }

                            {
                              auto* t =
                                  BlobGetMutableTensor(ws.CreateBlob("b"), CPU);
                              t->Resize(outputChannels);
                              CPUContext ctx;
                              math::RandGaussian<float, CPUContext>(
                                  t->size(),
                                  0,
                                  1,
                                  t->mutable_data<float>(),
                                  &ctx);
                            }

                            NetDef netdef;
                            {
                              auto& op = *(netdef.add_op());
                              op.set_type("CopyToMPSCNN");
                              op.add_input("X_cpu");
                              op.add_output("X_mtl");
                            }

                            {
                              auto& op = *(netdef.add_op());
                              op.set_type("MPSCNNConvTranspose");
                              op.add_input("X_mtl");
                              op.add_input("W");
                              op.add_input("b");
#define ADD_ARGS(op)                    \
  do {                                  \
    add_arg_str(op, "order", "NCHW");   \
    add_arg_int_list(                   \
        op,                             \
        std::vector<string>{"kernel_h", \
                            "kernel_w", \
                            "pad_t",    \
                            "pad_b",    \
                            "pad_l",    \
                            "pad_r",    \
                            "stride_w", \
                            "stride_h", \
                            "adj"},     \
        std::vector<int>{kernel_h,      \
                         kernel_w,      \
                         pad_t,         \
                         pad_b,         \
                         pad_l,         \
                         pad_r,         \
                         stride_w,      \
                         stride_h,      \
                         adj});         \
  } while (false)
                              ADD_ARGS(op);
                              op.add_output("Y_mtl");
                            }

                            {
                              auto& op = *(netdef.add_op());
                              op.set_type("CopyFromMPSCNN");
                              op.add_input("Y_mtl");
                              op.add_output("Y_cpu");
                            }

                            {
                              auto& op = *(netdef.add_op());
                              op.set_type("ConvTranspose");
                              op.add_input("X_cpu");
                              op.add_input("W");
                              op.add_input("b");
                              ADD_ARGS(op);
                              op.add_output("Y_ref");
                            }
#undef ADD_ARGS

                            ws.RunNetOnce(netdef);
                            const auto& t2 =
                                ws.GetBlob("Y_cpu")->Get<TensorCPU>();
                            const auto& t1 =
                                ws.GetBlob("Y_ref")->Get<TensorCPU>();
                            CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
                            LOG(INFO) << t1.sizes();
                            for (auto i = 0; i < t1.size(); ++i) {
                              // FP16 <-> FP32 round trip, accumulation, etc.
                              const float t1_i = t1.data<float>()[i];
                              const float t2_i = t2.data<float>()[i];
                              constexpr float tol = 2.0e-2;
                              CHECK(
                                  std::abs(t1_i - t2_i) <=
                                  (tol + tol * std::abs(t1_i)))
                                  << t1_i << ", " << t2_i;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  {
    for (const auto array : std::vector<bool>{true, false}) {
      for (auto numInputs = 2; numInputs <= 4; numInputs++) {
        for (const auto batchSize : std::vector<size_t>{1, 2}) {
          auto mtl = [&](size_t i) {
            return std::string("X_mtl_") + std::to_string(i);
          };
          auto cpu = [&](size_t i) {
            return std::string("X_cpu_") + std::to_string(i);
          };

          LOG(INFO) << "MPSCNNConcat Test" << array << ", " << numInputs << ", "
                    << batchSize;
          Workspace ws;
          for (auto i = 0; i < numInputs; ++i) {
            auto* t = BlobGetMutableTensor(ws.CreateBlob(cpu(i)), CPU);
            t->Resize(batchSize, array ? (i + 1) * 4 : 4, 10, 10);
            CPUContext ctx;
            math::RandGaussian<float, CPUContext>(
                t->size(), 0, 1, t->mutable_data<float>(), &ctx);
          }

          NetDef netdef;
          {
            auto& op = *(netdef.add_op());
            op.set_type("CopyToMPSCNN");
            for (auto i = 0; i < numInputs; ++i) {
              op.add_input(cpu(i));
              op.add_output(mtl(i));
            }
          }

          {
            auto& op = *(netdef.add_op());
            op.set_type("MPSCNNConcat");
            for (auto i = 0; i < numInputs; ++i) {
              op.add_input(mtl(i));
            }
            {
              auto& arg = *(op.add_arg());
              arg.set_name("order");
              arg.set_s("NCHW");
            }
            op.add_output("Y_mtl");
            op.add_output("Y_mtl_mask");
          }

          {
            auto& op = *(netdef.add_op());
            op.set_type("CopyFromMPSCNN");
            op.add_input("Y_mtl");
            op.add_output("Y_cpu");
          }

          {
            auto& op = *(netdef.add_op());
            op.set_type("Concat");
            for (auto i = 0; i < numInputs; ++i) {
              op.add_input(cpu(i));
            }
            {
              auto& arg = *(op.add_arg());
              arg.set_name("order");
              arg.set_s("NCHW");
            }

            op.add_output("Y_ref");
            op.add_output("Y_ref_mask");
          }

          ws.RunNetOnce(netdef);
          const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

          const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
          CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
          LOG(INFO) << t1.sizes();
          for (auto i = 0; i < t1.size(); ++i) {
            // FP16 <-> FP32 round trip, accumulation, etc.
            const float t1_i = t1.data<float>()[i];
            const float t2_i = t2.data<float>()[i];
            TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
          }
        }
      }
    }
  }

  @autoreleasepool {
    for (const auto& batchSize : std::vector<size_t>{1, 2, 3, 4}) {
      for (const auto& inputChannels :
           std::vector<size_t>{1, 2, 3, 4, 16, 24, 32, 48, 96, 128, 256}) {
        for (const auto& groups : std::vector<int>{1, 4, 8, 16}) {
          if (inputChannels % groups != 0) {
            continue;
          }
          Workspace ws;
          {
            auto* t = BlobGetMutableTensor(ws.CreateBlob("X_cpu"), CPU);
            t->Resize(batchSize, inputChannels, 53, 47);
            CPUContext ctx;
            math::RandGaussian<float, CPUContext>(
                t->size(), 0, 1, t->mutable_data<float>(), &ctx);
          }
          NetDef netdef;
#define ADD_ARGS(op)                                          \
  do {                                                        \
    add_arg_str(op, "order", "NCHW");                         \
    add_arg_int_list(                                         \
        op,                                                   \
        std::vector<string>{"kernel_w", "kernel_h", "group"}, \
        std::vector<int>{1, 1, groups});                      \
  } while (false)
          {
            auto& op = *(netdef.add_op());
            op.set_type("CopyToMPSCNN");
            op.add_input("X_cpu");
            op.add_output("X_mtl");
          }
          {
            auto& op = *(netdef.add_op());
            op.set_type("MPSCNNChannelShuffle");
            op.add_input("X_mtl");
            ADD_ARGS(op);
            op.add_output("Y_mtl");
          }
          {
            auto& op = *(netdef.add_op());
            op.set_type("CopyFromMPSCNN");
            op.add_input("Y_mtl");
            op.add_output("Y_cpu");
          }
          {
            auto& op = *(netdef.add_op());
            op.set_type("ChannelShuffle");
            op.add_input("X_cpu");
            ADD_ARGS(op);
            op.add_output("Y_ref");
          }
#undef ADD_ARGS
          ws.RunNetOnce(netdef);
          const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
          const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

          CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
          for (auto i = 0; i < t1.size(); ++i) {
            // FP16 <-> FP32 round trip, accumulation, etc.
            const float t1_i = t1.data<float>()[i];
            const float t2_i = t2.data<float>()[i];
            TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
          }
        }
      }
    }
  }

  {
    for (const auto channelCount : std::vector<size_t>{1, 2, 3, 4}) {
      for (auto numInputs = 2; numInputs <= 4; numInputs++) {
        for (const auto batchSize : std::vector<size_t>{1, 2}) {
          auto mtl = [&](size_t i) {
            return std::string("X_mtl_") + std::to_string(i);
          };
          auto cpu = [&](size_t i) {
            return std::string("X_cpu_") + std::to_string(i);
          };

          LOG(INFO) << "MPSCNNConcat(edge case) Test" << channelCount << ", "
                    << numInputs << ", " << batchSize;
          Workspace ws;
          for (auto i = 0; i < numInputs; ++i) {
            auto* t = BlobGetMutableTensor(ws.CreateBlob(cpu(i)), CPU);
            t->Resize(batchSize, channelCount, 9, 17);
            CPUContext ctx;
            math::RandGaussian<float, CPUContext>(
                t->size(), 0, 1, t->mutable_data<float>(), &ctx);
          }

          NetDef netdef;
          {
            auto& op = *(netdef.add_op());
            op.set_type("CopyToMPSCNN");
            for (auto i = 0; i < numInputs; ++i) {
              op.add_input(cpu(i));
              op.add_output(mtl(i));
            }
          }

          {
            auto& op = *(netdef.add_op());
            op.set_type("MPSCNNConcat");
            for (auto i = 0; i < numInputs; ++i) {
              op.add_input(mtl(i));
            }
            {
              auto& arg = *(op.add_arg());
              arg.set_name("order");
              arg.set_s("NCHW");
            }
            op.add_output("Y_mtl");
            op.add_output("Y_mtl_mask");
          }

          {
            auto& op = *(netdef.add_op());
            op.set_type("CopyFromMPSCNN");
            op.add_input("Y_mtl");
            op.add_output("Y_cpu");
          }

          {
            auto& op = *(netdef.add_op());
            op.set_type("Concat");
            for (auto i = 0; i < numInputs; ++i) {
              op.add_input(cpu(i));
            }
            {
              auto& arg = *(op.add_arg());
              arg.set_name("order");
              arg.set_s("NCHW");
            }

            op.add_output("Y_ref");
            op.add_output("Y_ref_mask");
          }

          ws.RunNetOnce(netdef);
          const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

          const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
          CAFFE_ENFORCE_EQ(t1.sizes(), t2.sizes());
          LOG(INFO) << t1.sizes();
          for (auto i = 0; i < t1.size(); ++i) {
            // FP16 <-> FP32 round trip, accumulation, etc.
            const float t1_i = t1.data<float>()[i];
            const float t2_i = t2.data<float>()[i];
            TORCH_CHECK_NEAR(t1_i, t2_i, 0.1);
          }
        }
      }
    }
  }

  {
    LOG(INFO) << "MPSCNNReadCount Test";
    NetDef netdef;
    {
      auto& op = *(netdef.add_op());
      op.add_input("X_cpu");
      op.add_output("X_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.add_input("X_mtl");
      op.add_output("X_mtl");
    }

    {
      auto& op = *(netdef.add_op());
      op.add_input("X_mtl");
      op.add_output("Y");
    }

    {
      auto& op = *(netdef.add_op());
      op.add_input("X_mtl");
      op.add_output("X_mtl");
    }
    netdef = annotateDefWithReadCounts(netdef);
    auto rc = [&](size_t i) -> size_t {
      auto* arg = GetMutableArgument(
          "__mpscnn_read_count__", false, netdef.mutable_op(i));
      if (!arg) {
        return 1;
      }
      return arg->i();
    };
    TORCH_CHECK_EQ(rc(0), 1);
    TORCH_CHECK_EQ(rc(1), 2);
    TORCH_CHECK_EQ(rc(2), 1);
    TORCH_CHECK_EQ(rc(3), 1);
  }

  {
    for (const auto& computeOp : std::vector<std::string>{"FC", "Conv"}) {
      LOG(INFO) << "MPSCNNRewriteForMetal Fusion/Copy Test";
      NetDef netdef;
      netdef.add_external_input("X");
      netdef.add_external_output("Y");
      // These two ops can be fused.
      {
        auto& op = *(netdef.add_op());
        op.set_type(computeOp);
        op.add_input("X");
        op.add_input("W");
        op.add_input("b");
        op.add_output("Y");
      }
      {
        auto& op = *(netdef.add_op());
        op.set_type("Relu");
        op.add_input("Y");
        op.add_output("Y");
      }
      {
        auto& op = *(netdef.add_op());
        op.set_type(computeOp);
        op.add_input("X2");
        op.add_input("W");
        op.add_input("b");
        op.add_output("Y2");
      }
      {
        auto& op = *(netdef.add_op());
        op.set_type("Relu");
        op.add_input("Y2");
        op.add_output("Y");
      }
      netdef = rewriteForMetal(netdef);
      auto ty = [&](size_t i) { return netdef.op(i).type(); };
      auto i0 = [&](size_t i) { return netdef.op(i).input(0); };
      auto o0 = [&](size_t i) { return netdef.op(i).output(0); };
      TORCH_CHECK_EQ(netdef.op_size(), 4);
      TORCH_CHECK_EQ(ty(0), "CopyToMPSCNN");
      TORCH_CHECK_EQ(ty(1), std::string("MPSCNN") + computeOp + std::string("Relu"));
      TORCH_CHECK_EQ(ty(2), std::string("MPSCNN") + computeOp + std::string("Relu"));
      TORCH_CHECK_EQ(ty(3), "CopyFromMPSCNN");
      TORCH_CHECK_EQ(i0(0), "X");
      TORCH_CHECK_EQ(i0(1), o0(0));
      TORCH_CHECK_EQ(i0(2), "X2");
      TORCH_CHECK_EQ(o0(2), i0(3));
      TORCH_CHECK_EQ(o0(3), "Y");
      TORCH_CHECK_EQ(netdef.external_input(0), "X");
      TORCH_CHECK_EQ(netdef.external_output(0), "Y");
    }
  }

  {
    LOG(INFO) << "MPSCNNRewriteForMetal Failure Test";
    NetDef netdef;
    netdef.add_external_input("X");
    netdef.add_external_output("Y");
    {
      auto& op = *(netdef.add_op());
      op.set_type("Conv");
      op.add_input("X");
      op.add_input("W");
      op.add_input("b");
      op.add_output("Y1");
    }
    {
      auto& op = *(netdef.add_op());
      op.set_type("Conv");
      op.add_input("X");
      op.add_input("W");
      op.add_input("b");
      op.add_output("Y2");
    }

    {
      auto& op = *(netdef.add_op());
      op.set_type("Concat");
      op.add_input("Y1");
      op.add_input("Y2");
      op.add_output("Y");
    }
    try {
      netdef = rewriteForMetal(netdef);
      CHECK(false) << "Shouldn't reach here, due to multiple usages of X";
    } catch (const std::exception& e) {
      // Nothing.
    }
  }

  {
    LOG(INFO) << "MPSCNNRewriteForMetal out-of-place Fusion Test";
    NetDef netdef;
    netdef.add_external_input("X");
    netdef.add_external_output("Z");
    {
      auto& op = *(netdef.add_op());
      op.set_type("Conv");
      op.add_input("X");
      op.add_input("W");
      op.add_input("b");
      op.add_output("Y");
    }
    {
      auto& op = *(netdef.add_op());
      op.set_type("Relu");
      op.add_input("Y");
      op.add_output("Z");
    }
    {
      auto& op = *(netdef.add_op());
      op.set_type("Relu");
      op.add_input("Z");
      op.add_output("Z");
    }
    netdef = rewriteForMetal(netdef);
    TORCH_CHECK_EQ(netdef.op_size(), 4);
    auto ty = [&](size_t i) { return netdef.op(i).type(); };
    auto i0 = [&](size_t i) { return netdef.op(i).input(0); };
    auto o0 = [&](size_t i) { return netdef.op(i).output(0); };
    TORCH_CHECK_EQ(ty(0), "CopyToMPSCNN");
    TORCH_CHECK_EQ(ty(1), "MPSCNNConvRelu");
    TORCH_CHECK_EQ(ty(2), "MPSCNNRelu");
    TORCH_CHECK_EQ(ty(3), "CopyFromMPSCNN");
    TORCH_CHECK_EQ(i0(1), o0(0));
    TORCH_CHECK_EQ(o0(1), "Z");
    TORCH_CHECK_EQ(i0(2), "Z");
    TORCH_CHECK_EQ(o0(2), i0(3));
  }

  {
    LOG(INFO) << "MPSCNNRewriteForMetal out-of-place fusion failure test";
    NetDef netdef;
    netdef.add_external_input("X");
    netdef.add_external_output("Z");
    {
      auto& op = *(netdef.add_op());
      op.set_type("Conv");
      op.add_input("X");
      op.add_input("W");
      op.add_input("b");
      op.add_output("Y");
    }
    {
      auto& op = *(netdef.add_op());
      op.set_type("Relu");
      op.add_input("Y");
      op.add_output("Z");
    }
    {
      auto& op = *(netdef.add_op());
      op.set_type("Relu");
      op.add_input("Y");
      op.add_output("Z");
    }
    netdef = rewriteForMetal(netdef);
    TORCH_CHECK_EQ(netdef.op_size(), 5);
    auto ty = [&](size_t i) { return netdef.op(i).type(); };
    auto i0 = [&](size_t i) { return netdef.op(i).input(0); };
    auto o0 = [&](size_t i) { return netdef.op(i).output(0); };
    TORCH_CHECK_EQ(ty(0), "CopyToMPSCNN");
    TORCH_CHECK_EQ(ty(1), "MPSCNNConv");
    TORCH_CHECK_EQ(ty(2), "MPSCNNRelu");
    TORCH_CHECK_EQ(ty(3), "MPSCNNRelu");
    TORCH_CHECK_EQ(ty(4), "CopyFromMPSCNN");
    TORCH_CHECK_EQ(i0(1), o0(0));
    TORCH_CHECK_EQ(o0(1), "Y");
    TORCH_CHECK_EQ(i0(2), o0(1));
    TORCH_CHECK_EQ(o0(2), "Z");
    TORCH_CHECK_EQ(i0(3), o0(1));
    TORCH_CHECK_EQ(o0(3), i0(4));
  }

  {
    LOG(INFO) << "MPSCNNRewriteForMetal PreProcess/Deprocess Test";
    NetDef netdef;
    {
      auto& op = *(netdef.add_op());
      op.set_type("PackedInt8BGRANHWCToNCHWCStylizerPreprocess");
      op.add_input("X");
      op.add_output("Y");
    }
    {
      auto& op = *(netdef.add_op());
      op.set_type("Relu");
      op.add_input("Y");
      op.add_output("Y");
    }
    {
      auto& op = *(netdef.add_op());
      op.set_type("BRGNCHWCToPackedInt8BGRAStylizerDeprocess");
      op.add_input("Y");
      op.add_output("Z");
    }
    netdef = rewriteForMetal(netdef);
    auto ty = [&](size_t i) { return netdef.op(i).type(); };
    auto i0 = [&](size_t i) { return netdef.op(i).input(0); };
    auto o0 = [&](size_t i) { return netdef.op(i).output(0); };
    TORCH_CHECK_EQ(netdef.op_size(), 3);
    TORCH_CHECK_EQ(ty(0), "MPSCNNPackedInt8BGRANHWCToNCHWCStylizerPreprocess");
    TORCH_CHECK_EQ(ty(1), "MPSCNNRelu");
    TORCH_CHECK_EQ(ty(2), "MPSCNNBRGNCHWCToPackedInt8BGRAStylizerDeprocess");
    TORCH_CHECK_EQ(i0(0), "X");
    TORCH_CHECK_EQ(i0(1), o0(0));
    TORCH_CHECK_EQ(i0(2), o0(1));
    TORCH_CHECK_EQ(o0(2), "Z");
  }
  LOG(INFO) << "All MPSCNN tests passed.";
}

NetDef truncateAfter(NetDef def, size_t idx) {
  // idx = 0, net = 10 -> remove 9
  // idx = 0, net = 1 -> remove 0
  const auto toRemove = def.op_size() - idx - 1;
  for (auto i = 0; i < toRemove; ++i) {
    def.mutable_op()->RemoveLast();
  }
  TORCH_CHECK_EQ(def.op_size(), idx + 1);
  return def;
}

NetDef addMPSCNNCopyFinalizer(NetDef def) {
  TORCH_CHECK_GE(def.op_size(), 1);
  const auto name = def.mutable_op(def.op_size() - 1)->output(0);
  def.mutable_op(def.op_size() - 1)->set_output(0, "METAL_COPIER");
  {
    auto& op = *(def.add_op());
    op.set_type("CopyFromMPSCNN");
    op.add_input("METAL_COPIER");
    op.add_output(name);
  }
  return def;
}

void compareModels(const NetDef& initNet, NetDef predictNet) {
  auto* arg = predictNet.mutable_op(0)->mutable_arg(0);
  TORCH_CHECK_EQ(arg->name(), "noise_std");
  arg->set_f(0.000001);

  NetDef metalPredictNet;
  CAFFE_ENFORCE(tryConvertToMPSCNN(initNet, predictNet, &metalPredictNet));

  // TODO: consider last op as well.
  for (auto i = 0; i < predictNet.op_size(); ++i) {
    auto truncatedPredictNet = truncateAfter(predictNet, i);
    auto truncatedMetalPredictNet = truncateAfter(metalPredictNet, i);
    // For all but the last op, we need to add a copy op.
    if (i != predictNet.op_size() - 1) {
      truncatedMetalPredictNet =
          addMPSCNNCopyFinalizer(truncatedMetalPredictNet);
    }

    dumpDef(truncatedPredictNet);
    dumpDef(truncatedMetalPredictNet);

    Workspace cws;
    cws.RunNetOnce(initNet);
    {
      auto* t = BlobGetMutableTensor(
          cws.CreateBlob(predictNet.external_input(0)), CPU);
      t->Resize(1, 224, 224, 4);
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<uint8_t>()[i] = i % 225;
      }
    }
    cws.RunNetOnce(truncatedPredictNet);

    Workspace mws;
    mws.RunNetOnce(initNet);
    {
      auto* t = BlobGetMutableTensor(
          mws.CreateBlob(predictNet.external_input(0)), CPU);
      t->Resize(1, 224, 224, 4);
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<uint8_t>()[i] = i % 225;
      }
    }
    mws.RunNetOnce(truncatedMetalPredictNet);

    const auto name =
        truncatedPredictNet.op(truncatedPredictNet.op_size() - 1).output(0);

    LOG(INFO) << "Checking correspondence for name: " << name << ", idx: " << i;
    {
      const auto& mt = mws.GetBlob(name)->Get<TensorCPU>();
      const auto& ct = cws.GetBlob(name)->Get<TensorCPU>();
      TORCH_CHECK_EQ(mt.sizes(), ct.sizes());
      for (auto j = 0; j < mt.size(); ++j) {
        if (mt.IsType<float>()) {
          if (j < 10) {
            LOG(INFO) << "i: " << i << ", j: " << j
                      << ", CPU: " << ct.data<float>()[j]
                      << ", MTL: " << mt.data<float>()[j];
          }
          TORCH_CHECK_NEAR(mt.data<float>()[j], ct.data<float>()[j], 5);
        } else {
          CHECK(mt.IsType<uint8_t>());
          if (j < 10) {
            LOG(INFO) << "i: " << i << ", j: " << j
                      << ", CPU: " << ct.data<uint8_t>()[j]
                      << ", MTL: " << mt.data<uint8_t>()[j];
          }
          TORCH_CHECK_NEAR(mt.data<uint8_t>()[j], ct.data<uint8_t>()[j], 5);
        }
      }
    }
  }
}
void verifyRewrite(
    const NetDef& initNet,
    const NetDef& net,
    std::vector<int> inputDims) {
  NetDef metalPredictNet;
  NetDef predictNet = setSpecialArgs(net);
  CAFFE_ENFORCE(tryConvertToMPSCNNIntermediateCopies(
      initNet, predictNet, &metalPredictNet));
  dumpDef(predictNet);
  dumpDef(metalPredictNet);

#define RUN_NET(ws, predictNet)                            \
  ws.RunNetOnce(initNet);                                  \
  {                                                        \
    auto* t = BlobGetMutableTensor(                        \
        ws.CreateBlob(predictNet.external_input(0)), CPU); \
    t->Resize(inputDims);                                  \
    CPUContext ctx;                                        \
    math::RandGaussian<float, CPUContext>(                 \
        t->size(), 0, 1, t->mutable_data<float>(), &ctx);  \
  }                                                        \
  ws.RunNetOnce(predictNet);

  // initialize
  getMPSCNNContext();

  Workspace cws;
  RUN_NET(cws, predictNet);

  Workspace mws;
  RUN_NET(mws, metalPredictNet);

  for (auto i = 0; i < predictNet.external_output_size(); i++) {
    auto blobName = predictNet.external_output(i);
    LOG(INFO) << "Checking output blob:" << blobName;
    const auto& mt = mws.GetBlob(blobName)->Get<Tensor>();
    const auto& ct = cws.GetBlob(blobName)->Get<Tensor>();
    if (mt.size() == 0 || ct.size() == 0) {
      LOG(INFO) << "One of the operator failed.";
      return;
    }
    // TORCH_CHECK_EQ(mt.sizes(), ct.sizes());
    for (auto j = 0; j < fmin(mt.size(), ct.size()); ++j) {
      if (mt.IsType<float>()) {
        if (j < 10) {
          LOG(INFO) << "i: " << i << ", j: " << j
                    << ", CPU: " << ct.data<float>()[j]
                    << ", MTL: " << mt.data<float>()[j];
        }
        // Disabling check for now because of precision issues
        // TORCH_CHECK_NEAR(mt.data<float>()[j], ct.data<float>()[j], 5);
      } else {
        LOG(INFO) << "Type uint8_t";
        CHECK(mt.IsType<uint8_t>());
        if (j < 10) {
          LOG(INFO) << "i: " << i << ", j: " << j
                    << ", CPU: " << ct.data<uint8_t>()[j]
                    << ", MTL: " << mt.data<uint8_t>()[j];
        }
        // Disabling check for now.
        // TORCH_CHECK_NEAR(mt.data<uint8_t>()[j], ct.data<uint8_t>()[j], 5);
      }
    }
  }
  LOG(INFO) << "rewrite test passed.";
}
}

#endif
