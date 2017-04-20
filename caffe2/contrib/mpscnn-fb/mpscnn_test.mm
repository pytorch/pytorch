// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/core/common.h"

#if CAFFE2_MOBILE

#include "mpscnn.h"
#include "mpscnn_context.h"

#include "caffe2/core/logging.h"
#include "caffe2/core/workspace.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

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
              auto mtl = [&](size_t i) { return std::string("X_mtl_") + std::to_string(i); };
              auto cpu = [&](size_t i) { return std::string("X_cpu_") + std::to_string(i); };
              auto y_cpu = [&](size_t i) { return std::string("Y_cpu_") + std::to_string(i); };

              Workspace ws;
              for (auto i = 0; i < N; ++i) {
                auto* t = ws.CreateBlob(cpu(i))->GetMutable<TensorCPU>();
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
                CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
                for (auto i = 0; i < t1.size(); ++i) {
                  // FP16 <-> FP32 round trip.
                  CHECK_NEAR(t1.data<float>()[i], t2.data<float>()[i], 1e-2);
                }
              }
            }
          }
        }
      }
    }
  }

  {
    LOG(INFO) << "MPSCNNInstanceNorm Test";
    Workspace ws;
    {
      auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
      t->Resize(1, 3, 120, 140);
      CPUContext ctx;
      // Too noisy.
      math::RandGaussian<float, CPUContext>(t->size(), 0, 3, t->mutable_data<float>(), &ctx);
    }

    {
      auto* t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
      t->Resize(3);
      CPUContext ctx;
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<float>()[i] = i;
      }
      // Too noisy.
      // math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }
    {
      auto* t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
      t->Resize(3);
      CPUContext ctx;
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<float>()[i] = 8 - 2 * i;
      }
      // Too noisy.
      // math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
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
      op.set_type("MPSCNNInstanceNorm");
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
      op.set_type("InstanceNorm");
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

    CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      CHECK_NEAR(t1_i, t2_i, 0.05);
    }
  }

  {
    LOG(INFO) << "MPSCNNInstanceNorm Test";
    Workspace ws;
    {
      auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
      t->Resize(1, 12, 120, 140);
      CPUContext ctx;
      // Too noisy.
      math::RandGaussian<float, CPUContext>(t->size(), 0, 3, t->mutable_data<float>(), &ctx);
    }

    {
      auto* t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
      t->Resize(12);
      CPUContext ctx;
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<float>()[i] = i;
      }
      // Too noisy.
      // math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }
    {
      auto* t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
      t->Resize(12);
      CPUContext ctx;
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<float>()[i] = 8 - 2 * i;
      }
      // Too noisy.
      // math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
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
      op.set_type("MPSCNNInstanceNorm");
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
      op.set_type("InstanceNorm");
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

    CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      CHECK_NEAR(t1_i, t2_i, 0.1);
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
            auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
            t->Resize(batch_size, channels, 8, 13);
            CPUContext ctx;
            math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
          }

          {
            auto* t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
            t->Resize(shared ? channels : 1);
            CPUContext ctx;
            math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
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

          CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
          for (auto i = 0; i < t1.size(); ++i) {
            // FP16 <-> FP32 round trip, accumulation, etc.
            const float t1_i = t1.data<float>()[i];
            const float t2_i = t2.data<float>()[i];
            CHECK_NEAR(t1_i, t2_i, 0.1);
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
          auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
          t->Resize(batch_size, channels, 8, 13);
          CPUContext ctx;
          math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
        }

        for (const std::string name : {"scale", "bias", "mean", "var"}) {
          auto* t = ws.CreateBlob(name)->GetMutable<TensorCPU>();
          t->Resize(channels);
          CPUContext ctx;
          // High mean to avoid var division by zero.
          math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
          if (name == "var") {
            for (auto i = 0; i < t->size(); ++i) {
              t->mutable_data<float>()[i] = std::abs(t->mutable_data<float>()[i]) + 0.5;
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
            arg.set_name("is_test");
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
            arg.set_name("is_test");
            arg.set_i(1);
          }

          op.add_output("Y_ref");
        }

        ws.RunNetOnce(netdef);
        const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
        const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

        CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
        for (auto i = 0; i < t1.size(); ++i) {
          // FP16 <-> FP32 round trip, accumulation, etc.
          const float t1_i = t1.data<float>()[i];
          const float t2_i = t2.data<float>()[i];
          CHECK_NEAR(t1_i, t2_i, 0.1);
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
                auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
                t->Resize(batchSize, CIn, H, W);
                CPUContext ctx;
                math::RandGaussian<float, CPUContext>(
                    t->size(), 0, 1, t->mutable_data<float>(), &ctx);
              }

              {
                auto* t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
                t->Resize(COut, CIn * H * W);
                CPUContext ctx;
                math::RandGaussian<float, CPUContext>(
                    t->size(), 0, 1, t->mutable_data<float>(), &ctx);
              }

              {
                auto* t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
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
                  std::vector<TIndex>{TIndex(batchSize), TIndex(COut)});
              // Note dims do not match, as Metal leaves a 1x1 spatial dimension.
              CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());

              for (auto i = 0; i < t1.size(); ++i) {
                // FP16 <-> FP32 round trip, accumulation, etc.
                const float t1_i = t1.data<float>()[i];
                const float t2_i = t2.data<float>()[i];
                // LOG(INFO) << "i: " << i << ", cpu: " << t1_i << ", mtl: " << t2_i;
                CHECK_NEAR(t1_i, t2_i, 0.7);
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

          LOG(INFO) << "MPSCNNPool Test: " << pool;
          Workspace ws;
          {
            auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
            t->Resize(batchSize, 8, 8, 13);
            CPUContext ctx;
            math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
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
            op.set_type(std::string("MPSCNN") + pool);
            op.add_input("X_mtl");
            {
              auto& arg = *(op.add_arg());
              arg.set_name("kernel");
              arg.set_i(4);
            }
            {
              auto& arg = *(op.add_arg());
              arg.set_name("stride");
              arg.set_i(global_pooling ? 1 : 4);
            }
            {
              auto& arg = *(op.add_arg());
              arg.set_name("global_pooling");
              arg.set_i(global_pooling);
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
            op.set_type(pool);
            op.add_input("X_cpu");
            {
              auto& arg = *(op.add_arg());
              arg.set_name("kernel");
              arg.set_i(4);
            }
            {
              auto& arg = *(op.add_arg());
              arg.set_name("stride");
              arg.set_i(global_pooling ? 1 : 4);
            }
            {
              auto& arg = *(op.add_arg());
              arg.set_name("global_pooling");
              arg.set_i(global_pooling);
            }

            op.add_output("Y_ref");
          }

          ws.RunNetOnce(netdef);
          const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
          const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

          CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
          for (auto i = 0; i < t1.size(); ++i) {
            // FP16 <-> FP32 round trip, accumulation, etc.
            const float t1_i = t1.data<float>()[i];
            const float t2_i = t2.data<float>()[i];
            CHECK_NEAR(t1_i, t2_i, 0.1);
          }
        }
      }
    }
  }

  {
    LOG(INFO) << "MPSCNNPadImage Test";
    for (const auto dims : std::vector<std::vector<size_t>>{{1, 3, 50, 80}, {1, 12, 50, 80}}) {
      Workspace ws;
      {
        auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
        t->Resize(dims);
        CPUContext ctx;
        math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
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

      CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
      for (auto i = 0; i < t1.size(); ++i) {
        // FP16 <-> FP32 round trip, accumulation, etc.
        const float t1_i = t1.data<float>()[i];
        const float t2_i = t2.data<float>()[i];
        // LOG(INFO) << "i: " << i << ", " << "CPU: " << t1_i << ", MTL: " << t2_i;
        CHECK_NEAR(t1_i, t2_i, 0.01);
      }
    }
  }

  {
    LOG(INFO) << "MPSCNNPreprocess Test";
    Workspace ws;
    {
      auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
      t->Resize(1, 8, 13, 4);
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

    CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      CHECK_NEAR(t1_i, t2_i, 0.1);
    }
  }

  {
    LOG(INFO) << "MPSCNNDeprocess Test";
    Workspace ws;
    {
      auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
      t->Resize(1, 3, 8, 24);
      CPUContext ctx;
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<float>()[i] = rand() % 255;
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

    CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<uint8_t>()[i];
      const float t2_i = t2.data<uint8_t>()[i];
      CHECK_NEAR(t1_i, t2_i, 0.1);
    }
  }

  {
    LOG(INFO) << "MPSCNNDeprocess Test";
    Workspace ws;
    {
      auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
      t->Resize(1, 3, 1280, 720);
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

    CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<uint8_t>()[i];
      const float t2_i = t2.data<uint8_t>()[i];
      CHECK_NEAR(t1_i, t2_i, 0.1);
    }
  }

  {
    for (const auto& batchSize : std::vector<size_t>{1, 2}) {
      LOG(INFO) << "MPSCNNConv Test";
      Workspace ws;
      {
        auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
        t->Resize(batchSize, 12, 57, 72);
        CPUContext ctx;
        math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
      }

      {
        auto* t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
        t->Resize(8, 12, 3, 3);
        CPUContext ctx;
        math::RandGaussian<float, CPUContext>(8 * 12 * 3 * 3, 0, 1, t->mutable_data<float>(), &ctx);
      }

      {
        auto* t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
        t->Resize(8);
        CPUContext ctx;
        math::RandGaussian<float, CPUContext>(8, 0, 1, t->mutable_data<float>(), &ctx);
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

      ws.RunNetOnce(netdef);
      const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
      const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

      CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
      for (auto i = 0; i < t1.size(); ++i) {
        // FP16 <-> FP32 round trip, accumulation, etc.
        const float t1_i = t1.data<float>()[i];
        const float t2_i = t2.data<float>()[i];
        CHECK_NEAR(t1_i, t2_i, 0.1);
      }
    }
  }

  {
    LOG(INFO) << "MPSCNNConvRelu Test";
    Workspace ws;
    {
      auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
      t->Resize(1, 12, 57, 72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }

    {
      auto* t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
      t->Resize(8, 12, 3, 3);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(8 * 12 * 3 * 3, 0, 1, t->mutable_data<float>(), &ctx);
    }

    {
      auto* t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
      t->Resize(8);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(8, 0, 1, t->mutable_data<float>(), &ctx);
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

    CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      CHECK_NEAR(t1_i, t2_i, 0.1);
    }
  }

  {
    LOG(INFO) << "MPSConv Test";
    Workspace ws;
    {
      auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
      t->Resize(1, 12, 57, 72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }

    {
      auto* t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
      t->Resize(8, 12, 3, 3);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(8 * 12 * 3 * 3, 0, 1, t->mutable_data<float>(), &ctx);
    }

    {
      auto* t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
      t->Resize(8);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(8, 0, 1, t->mutable_data<float>(), &ctx);
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

    CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      CHECK_NEAR(t1_i, t2_i, 0.1);
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
                auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
                t->Resize(batchSize, C, 12, 16);
                CPUContext ctx;
                math::RandGaussian<float, CPUContext>(
                    t->size(), 0, 1, t->mutable_data<float>(), &ctx);
              }

              {
                auto* t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
                t->Resize(M, C, K, K);
                CPUContext ctx;
                math::RandGaussian<float, CPUContext>(
                    t->size(), 0, 1, t->mutable_data<float>(), &ctx);
              }

              {
                auto* t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
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

              CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
              for (auto i = 0; i < t1.size(); ++i) {
                // FP16 <-> FP32 round trip, accumulation, etc.
                const float t1_i = t1.data<float>()[i];
                const float t2_i = t2.data<float>()[i];
                CHECK_NEAR(t1_i, t2_i, 0.1);
              }
            }
          }
        }
      }
    }
  }

  {
    LOG(INFO) << "MPSAdd Test";
    Workspace ws;
    {
      auto* t = ws.CreateBlob("X0_cpu")->GetMutable<TensorCPU>();
      t->Resize(1, 12, 57, 72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }

    {
      auto* t = ws.CreateBlob("X1_cpu")->GetMutable<TensorCPU>();
      t->Resize(1, 12, 57, 72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
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

    CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      CHECK_NEAR(t1_i, t2_i, 0.01);
    }
  }

  {
    LOG(INFO) << "MPSAdd Test";
    Workspace ws;
    {
      auto* t = ws.CreateBlob("X0_cpu")->GetMutable<TensorCPU>();
      t->Resize(1, 12, 57, 72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
    }

    {
      auto* t = ws.CreateBlob("X1_cpu")->GetMutable<TensorCPU>();
      t->Resize(1, 12, 57, 72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
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

    CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      CHECK_NEAR(t1_i, t2_i, 0.05);
    }
  }

  {
    for (const auto& n : {"Relu", "Tanh", "Sigmoid"}) {
      LOG(INFO) << "MPSCNNNeuron Test: " << n;
      Workspace ws;
      {
        auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
        t->Resize(1, 4, 12, 12);
        CPUContext ctx;
        math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
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

      CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
      for (auto i = 0; i < t1.size(); ++i) {
        // FP16 <-> FP32 round trip, accumulation, etc.
        const float t1_i = t1.data<float>()[i];
        const float t2_i = t2.data<float>()[i];
        CHECK_NEAR(t1_i, t2_i, 0.02);
      }
    }
  }

  {
    LOG(INFO) << "MPSCNNDropout Test";
    Workspace ws;
    {
      auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
      t->Resize(1, 12, 57, 72);
      CPUContext ctx;
      math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
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
        arg.set_name("is_test");
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
        arg.set_name("is_test");
        arg.set_i(1);
      }
      op.add_output("Y_ref");
      op.add_output("Y_mask");
    }

    ws.RunNetOnce(netdef);
    const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
    const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();
    CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
    LOG(INFO) << t1.dims();
    for (auto i = 0; i < t1.size(); ++i) {
      // FP16 <-> FP32 round trip, accumulation, etc.
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      CHECK_NEAR(t1_i, t2_i, 0.1);
    }
  }

  {
    for (const auto scale : std::vector<float>{1.0, 2.0, 0.0625}) {
      for (const auto pool : std::vector<size_t>{1, 3, 7}) {

        LOG(INFO) << "MPSCNNRoIWarp Test";
        Workspace ws;
        {
          auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
          t->Resize(1, 8, 40, 40);
          CPUContext ctx;
          math::RandGaussian<float, CPUContext>(t->size(), 4, 2, t->mutable_data<float>(), &ctx);
        }
        {
          // Use the batch-first encoding (n, [bbox])
          auto* t = ws.CreateBlob("R")->GetMutable<TensorCPU>();
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

        CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
        LOG(INFO) << t1.dims();
        for (auto i = 0; i < t1.size(); ++i) {
          // FP16 <-> FP32 round trip, accumulation, etc.
          const float t1_i = t1.data<float>()[i];
          const float t2_i = t2.data<float>()[i];
          CHECK_NEAR(t1_i, t2_i, 0.1);
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
          auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
          t->Resize(1, 8, 40, 40);
          CPUContext ctx;
          math::RandGaussian<float, CPUContext>(t->size(), 4, 2, t->mutable_data<float>(), &ctx);
        }
        {
          auto* t = ws.CreateBlob("R")->GetMutable<TensorCPU>();
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

        CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
        LOG(INFO) << t1.dims();
        for (auto i = 0; i < t1.size(); ++i) {
          // FP16 <-> FP32 round trip, accumulation, etc.
          const float t1_i = t1.data<float>()[i];
          const float t2_i = t2.data<float>()[i];
          CHECK_NEAR(t1_i, t2_i, 0.1);
        }
      }
    }
  }
  {
    for (const auto& batchSize : std::vector<size_t>{1, 2}) {
      LOG(INFO) << "MPSCNNSoftmax Test";
      Workspace ws;
      {
        auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
        // Only works for spatial dimension of (1, 1) - weird.
        t->Resize(batchSize, 12, 1, 1);
        CPUContext ctx;
        math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
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
      CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
      LOG(INFO) << t1.dims();
      for (auto i = 0; i < t1.size(); ++i) {
        // FP16 <-> FP32 round trip, accumulation, etc.
        const float t1_i = t1.data<float>()[i];
        const float t2_i = t2.data<float>()[i];
        CHECK_NEAR(t1_i, t2_i, 0.1);
      }
    }
  }

  {
    for (const auto& kernel : std::vector<size_t>{1, 2, 4}) {
      for (const auto& stride : std::vector<size_t>{1, 2, 3}) {
        for (const auto& pad : std::vector<size_t>{0, 1, 2}) {
          for (const auto& inputChannels : std::vector<size_t>{3, 8}) {
            for (const auto& outputChannels : std::vector<size_t>{3, 8}) {
              for (const auto& batchSize : std::vector<size_t>{1, 2}) {
                for (const auto& adj : {0, 1, 2}) {
                  if (adj >= stride) {
                    continue;
                  }
                  LOG(INFO) << "MPSConvTranspose Test";
                  Workspace ws;
                  {
                    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
                    t->Resize(batchSize, inputChannels, 8, 12);
                    CPUContext ctx;
                    math::RandGaussian<float, CPUContext>(
                        t->size(), 0, 1, t->mutable_data<float>(), &ctx);
                  }

                  {
                    auto* t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
                    t->Resize(inputChannels, outputChannels, kernel, kernel);
                    CPUContext ctx;
                    math::RandGaussian<float, CPUContext>(
                        t->size(), 0, 1, t->mutable_data<float>(), &ctx);
                  }

                  {
                    auto* t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
                    t->Resize(outputChannels);
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
                    op.set_type("MPSCNNConvTranspose");
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
                      arg.set_i(kernel);
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
                    {
                      auto& arg = *(op.add_arg());
                      arg.set_name("adj");
                      arg.set_i(adj);
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
                    op.set_type("ConvTranspose");
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
                      arg.set_i(kernel);
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
                    {
                      auto& arg = *(op.add_arg());
                      arg.set_name("adj");
                      arg.set_i(adj);
                    }
                    op.add_output("Y_ref");
                  }

                  ws.RunNetOnce(netdef);
                  const auto& t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
                  const auto& t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();
                  CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
                  LOG(INFO) << t1.dims();
                  for (auto i = 0; i < t1.size(); ++i) {
                    // FP16 <-> FP32 round trip, accumulation, etc.
                    const float t1_i = t1.data<float>()[i];
                    const float t2_i = t2.data<float>()[i];
                    CHECK_NEAR(t1_i, t2_i, 0.3);
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

          auto mtl = [&](size_t i) { return std::string("X_mtl_") + std::to_string(i); };
          auto cpu = [&](size_t i) { return std::string("X_cpu_") + std::to_string(i); };

          LOG(INFO) << "MPSCNNConcat Test" << array << ", " << numInputs << ", " << batchSize;
          Workspace ws;
          for (auto i = 0; i < numInputs; ++i) {
            auto* t = ws.CreateBlob(cpu(i))->GetMutable<TensorCPU>();
            t->Resize(batchSize, array ? (i + 1) * 4 : 4, 10, 10);
            CPUContext ctx;
            math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
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
          CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
          LOG(INFO) << t1.dims();
          for (auto i = 0; i < t1.size(); ++i) {
            // FP16 <-> FP32 round trip, accumulation, etc.
            const float t1_i = t1.data<float>()[i];
            const float t2_i = t2.data<float>()[i];
            CHECK_NEAR(t1_i, t2_i, 0.1);
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
      auto* arg = GetMutableArgument("__mpscnn_read_count__", false, netdef.mutable_op(i));
      if (!arg) {
        return 1;
      }
      return arg->i();
    };
    CHECK_EQ(rc(0), 1);
    CHECK_EQ(rc(1), 2);
    CHECK_EQ(rc(2), 1);
    CHECK_EQ(rc(3), 1);
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
      // Can't fuse these as not in-place (can fix by using SSA).
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
      CHECK_EQ(netdef.op_size(), 5);
      CHECK_EQ(ty(0), "CopyToMPSCNN");
      CHECK_EQ(ty(1), std::string("MPSCNN") + computeOp + std::string("Relu"));
      CHECK_EQ(ty(2), std::string("MPSCNN") + computeOp);
      CHECK_EQ(ty(3), "MPSCNNRelu");
      CHECK_EQ(ty(4), "CopyFromMPSCNN");
      CHECK_EQ(i0(0), "X");
      CHECK_EQ(i0(1), o0(0));
      CHECK_EQ(o0(2), "Y2");
      CHECK_EQ(i0(3), o0(2));
      CHECK_EQ(i0(4), o0(3));
      CHECK_NE(o0(4), i0(4));
      CHECK_EQ(netdef.external_input(0), "X");
      CHECK_EQ(netdef.external_output(0), "Y");
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
    CHECK_EQ(netdef.op_size(), 3);
    CHECK_EQ(ty(0), "MPSCNNPackedInt8BGRANHWCToNCHWCStylizerPreprocess");
    CHECK_EQ(ty(1), "MPSCNNRelu");
    CHECK_EQ(ty(2), "MPSCNNBRGNCHWCToPackedInt8BGRAStylizerDeprocess");
    CHECK_EQ(i0(0), "X");
    CHECK_EQ(i0(1), o0(0));
    CHECK_EQ(i0(2), o0(1));
    CHECK_EQ(o0(2), "Z");
  }
}

NetDef truncateAfter(NetDef def, size_t idx) {
  // idx = 0, net = 10 -> remove 9
  // idx = 0, net = 1 -> remove 0
  const auto toRemove = def.op_size() - idx - 1;
  for (auto i = 0; i < toRemove; ++i) {
    def.mutable_op()->RemoveLast();
  }
  CHECK_EQ(def.op_size(), idx + 1);
  return def;
}

NetDef addMPSCNNCopyFinalizer(NetDef def) {
  CHECK_GE(def.op_size(), 1);
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
  CHECK_EQ(arg->name(), "noise_std");
  arg->set_f(0.000001);

  NetDef metalPredictNet;
  CAFFE_ENFORCE(tryConvertToMPSCNN(initNet, predictNet, &metalPredictNet));

  // TODO: consider last op as well.
  for (auto i = 0; i < predictNet.op_size(); ++i) {
    auto truncatedPredictNet = truncateAfter(predictNet, i);
    auto truncatedMetalPredictNet = truncateAfter(metalPredictNet, i);
    // For all but the last op, we need to add a copy op.
    if (i != predictNet.op_size() - 1) {
      truncatedMetalPredictNet = addMPSCNNCopyFinalizer(truncatedMetalPredictNet);
    }

    dumpDef(truncatedPredictNet);
    dumpDef(truncatedMetalPredictNet);

    Workspace cws;
    cws.RunNetOnce(initNet);
    {
      auto* t = cws.CreateBlob(predictNet.external_input(0))->GetMutable<TensorCPU>();
      t->Resize(1, 224, 224, 4);
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<uint8_t>()[i] = i % 225;
      }
    }
    cws.RunNetOnce(truncatedPredictNet);

    Workspace mws;
    mws.RunNetOnce(initNet);
    {
      auto* t = mws.CreateBlob(predictNet.external_input(0))->GetMutable<TensorCPU>();
      t->Resize(1, 224, 224, 4);
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<uint8_t>()[i] = i % 225;
      }
    }
    mws.RunNetOnce(truncatedMetalPredictNet);

    const auto name = truncatedPredictNet.op(truncatedPredictNet.op_size() - 1).output(0);

    LOG(INFO) << "Checking correspondence for name: " << name << ", idx: " << i;
    {
      const auto& mt = mws.GetBlob(name)->Get<TensorCPU>();
      const auto& ct = cws.GetBlob(name)->Get<TensorCPU>();
      CHECK_EQ(mt.dims(), ct.dims());
      for (auto j = 0; j < mt.size(); ++j) {
        if (mt.IsType<float>()) {
          if (j < 10) {
            LOG(INFO) << "i: " << i << ", j: " << j << ", CPU: " << ct.data<float>()[j]
                      << ", MTL: " << mt.data<float>()[j];
          }
          CHECK_NEAR(mt.data<float>()[j], ct.data<float>()[j], 5);
        } else {
          CHECK(mt.IsType<uint8_t>());
          if (j < 10) {
            LOG(INFO) << "i: " << i << ", j: " << j << ", CPU: " << ct.data<uint8_t>()[j]
                      << ", MTL: " << mt.data<uint8_t>()[j];
          }
          CHECK_NEAR(mt.data<uint8_t>()[j], ct.data<uint8_t>()[j], 5);
        }
      }
    }
  }
}
}

#endif
