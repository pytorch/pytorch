#include "ulp.h"
#include "ulp_neon.h"
#include "gtest/gtest.h"

namespace caffe2 {

void conv(const ConvArgs& args,
          const TensorCPU& X,
          const TensorCPU& W,
          const TensorCPU* b,
          TensorCPU* Y) {
  const auto N = X.dim32(0);
  const auto IH = X.dim32(1);
  const auto IW = X.dim32(2);
  const auto KH = W.dim32(1);
  const auto KW = W.dim32(2);
  const auto IC = W.dim32(3);
  Y->Resize(X.dim32(0),
            (X.dim32(1) - KH + args.pad_t + args.pad_b) / args.stride_h + 1,
            (X.dim32(2) - KW + args.pad_l + args.pad_r) / args.stride_w + 1,
            W.dim32(0));
  TORCH_CHECK_EQ(W.dim32(3), X.dim32(3));
  const auto OH = Y->dim32(1);
  const auto OW = Y->dim32(2);
  const auto OC = Y->dim32(3);

  const auto* Xdata = X.data<float>();
  const auto* Wdata = W.data<float>();
  auto* Ydata = Y->mutable_data<float>();
  for (auto n = 0; n < N; ++n) {
    for (auto oh = 0; oh < OH; ++oh) {
      for (auto ow = 0; ow < OW; ++ow) {
        for (auto oc = 0; oc < OC; ++oc) {
          float acc = b ? b->data<float>()[oc] : 0.0;
          for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
              for (int ic = 0; ic < IC; ++ic) {
                if (kh + args.stride_h * oh - args.pad_t < 0 ||
                    kh + args.stride_h * oh - args.pad_t >= IH ||
                    kw + args.stride_w * ow - args.pad_l < 0 ||
                    kw + args.stride_w * ow - args.pad_l >= IW) {
                  continue;
                }
                const auto x =
                    Xdata[ic + IC * (kw + args.stride_w * ow - args.pad_l) +
                          IC * IW * (kh + args.stride_h * oh - args.pad_t) + n * IC * IW * IH];
                const auto w = Wdata[ic + IC * kw + IC * KW * kh + IC * KW * KH * oc];
                acc += x * w;
              }
            }
          }
          Ydata[oc + OC * ow + OC * OW * oh + n * OC * OW * OH] = acc;
        }
      }
    }
  }
}

int randInt(int a, int b) {
  std::random_device rd;
  std::default_random_engine gen(rd());
  return std::uniform_int_distribution<int>(a, b)(gen);
}

TensorCPU genTensor11(std::vector<int64_t> shape) {
  Tensor r(CPU);
  r.Resize(shape);

  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<float> dis(0, 1);

  for (auto i = 0; i < r.size(); ++i) {
    r.mutable_data<float>()[i] = dis(gen) > 0.5 ? -1.0 : 1.0;
  };
  return r;
}

TensorCPU genTensorUniform11(std::vector<int64_t> shape) {
  Tensor r(CPU);
  r.Resize(shape);

  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<float> dis(-5.0, 5.0);

  for (auto i = 0; i < r.size(); ++i) {
    r.mutable_data<float>()[i] = dis(gen);
  };
  return r;
}

TensorCPU genTensor0123(std::vector<int64_t> shape) {
  Tensor r(CPU);
  r.Resize(shape);

  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<float> dis(0.1, 3.9);

  for (auto i = 0; i < r.size(); ++i) {
    r.mutable_data<float>()[i] = std::floor(dis(gen));
  };
  return r;
}

TEST(ULP, QPadZero) {
  ConvArgs args;
  args.pad_l = 1;
  args.pad_r = 1;
  args.pad_t = 1;
  args.pad_b = 1;

  const auto ICQ = 1;

  auto X = genTensor11({1, 10, 10, ICQ * 8});
  Tensor XQ(CPU), XQPad(CPU);
  signQuantize(X, &XQ);
  qpad_zero(args, XQ, &XQPad);

  EXPECT_EQ(XQ.dim32(0), XQPad.dim32(0));
  EXPECT_EQ(XQ.dim32(1), XQPad.dim32(1) - 2 * args.pad_l);
  EXPECT_EQ(XQ.dim32(2), XQPad.dim32(2) - 2 * args.pad_t);
  EXPECT_EQ(XQ.dim32(3), XQPad.dim32(3));
  EXPECT_EQ(XQ.dim32(3), ICQ);
  EXPECT_EQ(XQPad.dim32(3), ICQ);
  const auto* XQdata = XQ.data<uint8_t>();
  const auto* XQPaddata = XQPad.data<uint8_t>();
  for (auto oh = 0; oh < XQPad.dim32(1); ++oh) {
    for (auto ow = 0; ow < XQPad.dim32(2); ++ow) {
      for (auto icq = 0; icq < ICQ; ++icq) {
        auto ih = oh - args.pad_l;
        auto iw = ow - args.pad_t;
        if (ih < 0 || ih >= XQ.dim32(1) || iw < 0 || iw >= XQ.dim32(2)) {
          EXPECT_EQ(XQPaddata[icq + ICQ * ow + ICQ * XQPad.dim32(2) * oh], 0);
        } else {
          EXPECT_EQ(XQPaddata[icq + ICQ * ow + ICQ * XQPad.dim32(2) * oh],
                    XQdata[icq + ICQ * iw + ICQ * XQ.dim32(2) * ih]);
        }
      }
    }
  }
}

inline void gemmNT(int M, int N, int K, const float* A, const float* B, float* C) {
  for (auto m = 0; m < M; ++m) {
    for (auto n = 0; n < N; ++n) {
      float acc = 0.0;
      for (auto k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[n * K + k];
      }
      C[m * N + n] = acc;
    }
  }
}

inline void qgemmNT(int M, int N, int K, const uint8_t* A, const uint8_t* B, float* C) {
  TORCH_CHECK_EQ(K % 8, 0);
  const int QK = K / 8;
  for (auto m = 0; m < M; ++m) {
    for (auto n = 0; n < N; ++n) {
      float acc = 0.0;
      for (auto qk = 0; qk < QK; ++qk) {
        uint8_t mk = A[m * QK + qk];
        uint8_t nk = B[n * QK + qk];
        auto cnt = __builtin_popcount(mk ^ nk);
        acc += cnt;
      }
      C[m * N + n] = K - 2 * acc;
    }
  }
}

void gemmTest(int64_t M, int64_t N, int64_t K) {
  auto X = genTensor11({M, K});
  auto W = genTensor11({N, K});
  Tensor XQ(CPU), WQ(CPU), YQ(CPU), Y(CPU);
  {
    signQuantize(X, &XQ);
    signQuantize(W, &WQ);
    YQ.Resize(M, N);
    qgemmNT(M, N, K, XQ.data<uint8_t>(), WQ.data<uint8_t>(), YQ.mutable_data<float>());
  }
  {
    Y.Resize(M, N);
    gemmNT(M, N, K, X.data<float>(), W.data<float>(), Y.mutable_data<float>());
  }
  EXPECT_TRUE(Y.sizes() == YQ.sizes());
  for (auto i = 0; i < Y.size(); ++i) {
    EXPECT_NEAR(Y.data<float>()[i], YQ.data<float>()[i], 1e-3);
  }
}

TEST(QConv, GemmTest) {
  gemmTest(8, 64, 64);
  gemmTest(16, 64, 256);
  gemmTest(24, 128, 192);
  gemmTest(32, 64, 64);
  gemmTest(40, 64, 128);
  gemmTest(64, 64, 256);
}

TEST(QConv, ConvTest) {
  int S = 9;
  int IC = 16;
  int OC = 28;
  int K = 3;
  auto X = genTensor11({1, S, S, IC});
  auto W = genTensor11({OC, K, K, IC});
  Tensor XQ(CPU), WQ(CPU), YQ(CPU), Y(CPU);
  {
    signQuantize(X, &XQ);
    signQuantize(W, &WQ);
    qconv(ConvArgs{}, XQ, WQ, nullptr, &YQ);
  }
  { conv(ConvArgs{}, X, W, nullptr, &Y); }
  EXPECT_TRUE(Y.sizes() == YQ.sizes());
  for (auto i = 0; i < Y.size(); ++i) {
    EXPECT_NEAR(Y.data<float>()[i], YQ.data<float>()[i], 1e-3);
  }
}

void ConvTest2b1b(int IC, int KH, int KW, int H, int W, int OC, int N, ConvArgs args) {
  args.stride_h = std::min(args.stride_h, KH);
  args.stride_w = std::min(args.stride_w, KW);
  args.pad_l = std::min(args.pad_l, KW - 1);
  args.pad_r = std::min(args.pad_r, KW - 1);
  args.pad_t = std::min(args.pad_t, KH - 1);
  args.pad_b = std::min(args.pad_b, KH - 1);

  LOG(INFO) << "IC: " << IC << ", KH: " << KH << ", KW: " << KW << ", H: " << H << ", W: " << W
            << ", OC: " << OC << ", N: " << N << ", pad_l: " << args.pad_l
            << ", pad_r: " << args.pad_r << ", pad_t: " << args.pad_t << ", pad_b: " << args.pad_b
            << ", stride_h: " << args.stride_h << ", stride_w: " << args.stride_w;
  auto X = genTensor0123({N, H, W, IC});
  auto W_ = genTensor11({OC, KH, KW, IC});
  auto bias = genTensorUniform11({OC});
  Tensor Y(CPU), YQ(CPU), Y2b1b(CPU), YOP(CPU);

  {
    std::vector<std::unique_ptr<TensorCPU>> XQs(k2b1bXBits);
    std::vector<std::unique_ptr<TensorCPU>> YQs(k2b1bXBits);
    for (auto i = 0; i < k2b1bXBits; ++i) {
      XQs[i] = std::make_unique<Tensor>(CPU);
      YQs[i] = std::make_unique<Tensor>(CPU);
    }
    Tensor WQN(CPU), WQ(CPU);
    uniformQuantize2b1b(X, XQs, 0.5, 1.0);
    signQuantize(W_, &WQ);
    filterNormalization11(WQ, &WQN);
    for (auto i = 0; i < XQs.size(); ++i) {
      qconv(args, *(XQs[i]), WQ, nullptr, YQs[i].get());
    }
    YQ.ResizeLike(*YQs[0]);
    const auto F = WQ.dim(0);
    const auto N = YQ.size() / F;

    run2b1bUnification(nullptr,
                       N,
                       F,
                       WQN.data<float>(),
                       YQs[0]->data<float>(),
                       YQs[1]->data<float>(),
                       F,
                       YQ.mutable_data<float>(),
                       F,
                       bias.data<float>());
  }

  {
    Workspace ws;
    auto state = create2b1bConvState(&ws, W_, &bias);
    run2b1bConvGeneric(state.get(), args, X, &Y2b1b);
  }
  {
    Workspace ws;
    OperatorDef def;
    def.set_type("QConv");
    def.add_input("X");
    def.add_input("W");
    def.add_input("b");
    def.add_output("Y");
    def.add_arg()->CopyFrom(MakeArgument("kernel_h", KH));
    def.add_arg()->CopyFrom(MakeArgument("order", std::string("NHWC")));
    def.add_arg()->CopyFrom(MakeArgument("kernel_w", KW));
    def.add_arg()->CopyFrom(MakeArgument("stride_h", args.stride_h));
    def.add_arg()->CopyFrom(MakeArgument("stride_w", args.stride_w));
    def.add_arg()->CopyFrom(MakeArgument("pad_l", args.pad_l));
    def.add_arg()->CopyFrom(MakeArgument("pad_r", args.pad_r));
    def.add_arg()->CopyFrom(MakeArgument("pad_t", args.pad_t));
    def.add_arg()->CopyFrom(MakeArgument("pad_b", args.pad_b));
    auto* Xws = BlobGetMutableTensor(ws.CreateBlob("X"), CPU);
    Xws->ResizeLike(X);
    Xws->ShareExternalPointer(X.mutable_data<float>(), X.size());
    auto* Wws = BlobGetMutableTensor(ws.CreateBlob("W"), CPU);
    Wws->ResizeLike(W_);
    Wws->ShareExternalPointer(W_.mutable_data<float>(), W_.size());
    auto* bws = BlobGetMutableTensor(ws.CreateBlob("b"), CPU);
    bws->ResizeLike(bias);
    bws->ShareExternalPointer(bias.mutable_data<float>(), bias.size());
    ws.RunOperatorOnce(def);
    YOP.CopyFrom(ws.GetBlob("Y")->Get<TensorCPU>());
  }

  { conv(args, X, W_, &bias, &Y); }

  EXPECT_TRUE(Y.sizes() == YQ.sizes());
  EXPECT_TRUE(Y.sizes() == Y2b1b.sizes());
  EXPECT_TRUE(Y.sizes() == YOP.sizes());

  // for (auto i = 0; i < Y.size(); ++i) {
  //   LOG(INFO) << "i: " << i << ", y[i]: " << Y.data<float>()[i]
  //             << ", y2b1b[i]: " << Y2b1b.data<float>()[i] << ", yq[i]: " << YQ.data<float>()[i];
  // }

  for (auto i = 0; i < Y.size(); ++i) {
    EXPECT_NEAR(Y.data<float>()[i], YQ.data<float>()[i], 1e-3);
  }

  for (auto i = 0; i < Y.size(); ++i) {
    EXPECT_NEAR(Y.data<float>()[i], Y2b1b.data<float>()[i], 1e-3);
  }

  for (auto i = 0; i < Y.size(); ++i) {
    EXPECT_NEAR(Y.data<float>()[i], YOP.data<float>()[i], 1e-3);
  }
}

ConvArgs ca(size_t pad = 0, size_t stride = 1) {
  ConvArgs r;
  r.pad_l = pad;
  r.pad_r = pad;
  r.pad_t = pad;
  r.pad_b = pad;
  r.stride_w = stride;
  r.stride_h = stride;
  return r;
}

TEST(QConv, 2b1bConvTest) {
  ConvTest2b1b(40, 3, 4, 10, 10, 32, 1, ca());
  ConvTest2b1b(59, 1, 1, 1, 1, 1, 1, ca());
  ConvTest2b1b(59, 2, 2, 3, 3, 1, 1, ca());
  ConvTest2b1b(59, 2, 2, 3, 3, 64, 1, ca());
  ConvTest2b1b(64, 1, 1, 1, 1, 1, 1, ca());
  ConvTest2b1b(64, 1, 1, 1, 1, 64, 1, ca());
  ConvTest2b1b(64, 2, 2, 3, 3, 1, 1, ca());
  ConvTest2b1b(64, 1, 1, 3, 3, 1, 1, ca());
  ConvTest2b1b(128, 1, 1, 1, 1, 128, 1, ca());
  ConvTest2b1b(128, 1, 1, 8, 8, 8, 1, ca());
  ConvTest2b1b(128, 3, 3, 25, 100, 16, 1, ca());
  ConvTest2b1b(64, 3, 3, 10, 10, 8, 1, ca());
  ConvTest2b1b(128, 1, 3, 10, 10, 16, 1, ca());
  ConvTest2b1b(256, 3, 3, 14, 17, 128, 1, ca());
  ConvTest2b1b(512, 3, 3, 3, 3, 3, 1, ca());
  ConvTest2b1b(64, 5, 5, 14, 17, 15, 1, ca(1, 2));
  ConvTest2b1b(64, 1, 3, 14, 17, 32, 1, ca());
  ConvTest2b1b(64, 2, 1, 14, 17, 7, 1, ca());
  ConvTest2b1b(128, 1, 1, 14, 17, 128, 1, ca());
  ConvTest2b1b(128, 1, 1, 14, 17, 32, 1, ca());
}

TEST(QConv, 2b1bInputPackingTest) {
  ConvTest2b1b(64, 1, 1, 1, 1, 128, 1, ca());
  ConvTest2b1b(8, 1, 1, 1, 1, 1, 1, ca());
  ConvTest2b1b(2, 1, 1, 1, 1, 1, 1, ca());
  ConvTest2b1b(2, 1, 1, 3, 3, 1, 1, ca());
  ConvTest2b1b(2, 2, 2, 3, 3, 1, 1, ca());
}

TEST(QConv, 2b1bConvTestRandomized) {
  auto rca = []() {
    ConvArgs r;
    r.pad_l = randInt(0, 3);
    r.pad_r = randInt(0, 3);
    r.pad_t = randInt(0, 3);
    r.pad_b = randInt(0, 3);
    r.stride_w = randInt(1, 3);
    r.stride_h = randInt(1, 3);
    return r;
  };
  for (auto i = 0; i < 10; ++i) {
    ConvTest2b1b(randInt(1, 64) * 8,
                 randInt(1, 4),
                 randInt(1, 4),
                 randInt(5, 12),
                 randInt(5, 12),
                 randInt(1, 64),
                 randInt(1, 2),
                 rca());
    // Test 3x3 path.
    ConvTest2b1b(randInt(1, 64) * 8,
                 3,
                 3,
                 randInt(5, 12),
                 randInt(5, 12),
                 randInt(1, 64),
                 randInt(1, 2),
                 rca());

    // Test 3x3s2 path.
    ConvTest2b1b(randInt(1, 64) * 8,
                 3,
                 3,
                 randInt(5, 12),
                 randInt(5, 12),
                 randInt(1, 64),
                 randInt(1, 2),
                 rca());
    // Test 3x3 path with packing.
    ConvTest2b1b(randInt(1, 64) * 8,
                 3,
                 3,
                 randInt(5, 12),
                 randInt(5, 12),
                 randInt(1, 8) * kGEMMTileSize,
                 randInt(1, 2),
                 rca());
    // Test 1x1 path
    ConvTest2b1b(randInt(1, 64) * 8,
                 1,
                 1,
                 randInt(5, 12),
                 randInt(5, 12),
                 randInt(1, 64),
                 randInt(1, 2),
                 ca());

    // Test 1x1 with direct packing
    ConvTest2b1b(randInt(1, 64) * 8,
                 1,
                 1,
                 randInt(5, 12),
                 randInt(5, 12),
                 randInt(1, 4) * kGEMMTileSize,
                 randInt(1, 2),
                 ca());
    // Entirely arbitrary, no padding codepath.
    ConvTest2b1b(randInt(1, 64) * 8,
                 randInt(1, 4),
                 randInt(1, 4),
                 randInt(5, 12),
                 randInt(5, 12),
                 randInt(1, 128),
                 randInt(1, 2),
                 rca());
    // Entirely arbitrary, mixed codepath.
    ConvTest2b1b(randInt(1, 64),
                 randInt(1, 4),
                 randInt(1, 4),
                 randInt(5, 12),
                 randInt(5, 12),
                 randInt(1, 128),
                 randInt(1, 2),
                 rca());
  }
}

} // namespace caffe2
