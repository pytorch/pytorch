#include "ulp.h"

#include <cstring>
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/eigen_utils.h"
#include "ulp_neon.h"

namespace caffe2 {

void uniformQuantize2b1b(const TensorCPU& X,
                         const std::vector<std::unique_ptr<TensorCPU>>& XQ,
                         float offset,
                         float inter_center_distance) {
  CAFFE_ENFORCE_GT(X.ndim(), 1);
  const auto N = X.size_to_dim(X.ndim() - 1);
  auto C = X.size() / N;
  const auto QC = divRoundUp(C,  8);
  auto XQs = X.dims().vec();
  XQs[X.ndim() - 1] = QC;
  CAFFE_ENFORCE_EQ(XQ.size(), k2b1bXBits);
  for (auto i = 0; i < k2b1bXBits; ++i) {
    XQ[i]->Resize(XQs);
  }
  const float* Xdata = X.data<float>();
  std::array<uint8_t*, k2b1bXBits> XQdata;
  for (auto i = 0; i < k2b1bXBits; ++i) {
    XQdata[i] = XQ[i]->mutable_data<uint8_t>();
  }
  for (auto n = 0; n < N; ++n) {
    for (auto qc = 0; qc < QC; ++qc) {
      // compute the block in X.
      std::array<uint8_t, k2b1bXBits> p = {{0, 0}};
      for (auto b = 0; b < 8; ++b) {
        const auto c = qc * 8 + b;
        if (c < C) {
          float v = Xdata[qc * 8 + b + C * n];
          if (v < offset) {
            // zero'd already.
          } else if (v < offset + inter_center_distance) {
            p[0] |= 1 << b;
          } else if (v < offset + 2 * inter_center_distance) {
            p[1] |= 1 << b;
          } else {
            p[0] |= 1 << b;
            p[1] |= 1 << b;
          }
        }
      }
      for (auto i = 0; i < k2b1bXBits; ++i) {
        XQdata[i][qc + QC * n] = p[i];
      }
    }
  }
}

void qconv(const ConvArgs& args,
           const TensorCPU& X,
           const TensorCPU& W,
           const TensorCPU* b,
           TensorCPU* Y) {
  const auto N = X.dim32(0);
  const auto IH = X.dim32(1);
  const auto IW = X.dim32(2);
  const auto KH = W.dim32(1);
  const auto KW = W.dim32(2);
  const auto KC = W.dim32(3);
  Y->Resize(X.dim32(0),
            (X.dim32(1) - KH + args.pad_t + args.pad_b) / args.stride_h + 1,
            (X.dim32(2) - KW + args.pad_l + args.pad_r) / args.stride_w + 1,
            W.dim32(0));
  const auto OH = Y->dim32(1);
  const auto OW = Y->dim32(2);
  const auto OC = Y->dim32(3);

  CAFFE_ENFORCE_EQ(W.dim32(3), X.dim32(3));

  const auto* Xdata = X.data<uint8_t>();
  const auto* Wdata = W.data<uint8_t>();
  auto* Ydata = Y->mutable_data<float>();
  for (size_t n = 0; n < N; ++n) {
    for (size_t oh = 0; oh < OH; ++oh) {
      for (size_t ow = 0; ow < OW; ++ow) {
        for (size_t oc = 0; oc < OC; ++oc) {
          float acc = 0.0;
          for (size_t kh = 0; kh < KH; ++kh) {
            const int32_t ih = (int32_t)kh + (int32_t)args.stride_h * oh - (int32_t)args.pad_t;
            for (size_t kw = 0; kw < KW; ++kw) {
              const int32_t iw = (int32_t)kw + (int32_t)args.stride_w * ow - (int32_t)args.pad_l;
              for (size_t kc = 0; kc < KC; ++kc) {
                const uint8_t w = Wdata[kc + KC * kw + KC * KW * kh + KC * KW * KH * oc];
                // Use unsigned integer math to avoid multiple comparisons (>= H, < 0).
                if ((size_t)ih >= (size_t)IH || (size_t)iw >= (size_t)IW) {
                  acc += __builtin_popcount(0 ^ w);
                } else {
                  const uint8_t x =
                      Xdata[kc + KC * (size_t)iw + KC * IW * (size_t)ih + n * KC * IW * IH];
                  const uint8_t w = Wdata[kc + KC * kw + KC * KW * kh + KC * KW * KH * oc];
                  acc += __builtin_popcount(x ^ w);
                }
              }
            }
          }
          Ydata[oc + OC * ow + OC * OW * oh + n * OC * OW * OH] =
              KW * KH * KC * 8 - 2 * acc + (b ? b->data<float>()[oc] : 0.0);
          ;
        }
      }
    }
  }
}

void qpad_zero(const ConvArgs& args, const TensorCPU& X, TensorCPU* Y) {
  CAFFE_ENFORCE_EQ(args.stride_h, 1);
  CAFFE_ENFORCE_EQ(args.stride_w, 1);
  const auto* Xdata = X.data<uint8_t>();
  Y->Resize(X.dim32(0),
            X.dim32(1) + args.pad_t + args.pad_b,
            X.dim32(2) + args.pad_l + args.pad_r,
            X.dim32(3));
  auto* Ydata = Y->mutable_data<uint8_t>();
  ::memset(Ydata, Y->nbytes(), 0);
  const auto C = Y->dim32(3);
  const auto XrowSize = X.dim32(3) * X.dim32(2);
  const auto YrowSize = Y->dim32(3) * Y->dim32(2);
  math::CopyMatrix<CPUContext>(1,
                               X.dim32(1),
                               XrowSize,
                               Xdata,
                               XrowSize,
                               Ydata + C * args.pad_l + YrowSize * args.pad_t,
                               YrowSize,
                               nullptr);
}

void signQuantize(const TensorCPU& X, TensorCPU* XQ) {
  CAFFE_ENFORCE_GT(X.ndim(), 1);
  const auto N = X.size_to_dim(X.ndim() - 1);
  auto C = X.size() / N;
  const auto QC = divRoundUp(C,  8);
  auto XQs = X.dims().vec();
  XQs[X.ndim() - 1] = QC;
  XQ->Resize(XQs);
  const float* Xdata = X.data<float>();
  uint8_t* XQdata = XQ->mutable_data<uint8_t>();
  for (auto n = 0; n < N; ++n) {
    for (auto qc = 0; qc < QC; ++qc) {
      // compute the block in X.
      uint8_t p = 0;
      for (auto b = 0; b < 8; ++b) {
        const auto c = qc * 8 + b;
        if (c < C) {
          p |= (Xdata[c + C * n] > 0) << b;
        }
      }
      XQdata[qc + QC * n] = p;
    }
  }
}

void filterNormalization11(const TensorCPU& WQ, TensorCPU* WQN) {
  const auto F = WQ.dim32(0);
  // In our NEON kernel we read up to TileSize, so align allocation to TileSize elements.
  WQN->Resize(divRoundUp(F, kGEMMTileSize) * kGEMMTileSize);
  const auto WQs = WQ.size() / F;
  const auto WQbits = 8 * WQs;
  const auto* WQdata = WQ.data<uint8_t>();
  auto* WQNdata = WQN->mutable_data<float>();
  for (auto f = 0; f < F; ++f) {
    int32_t bitSum = 0;
    for (auto j = 0; j < WQs; ++j) {
      bitSum += __builtin_popcount(WQdata[f * WQs + j]);
    }
    DCHECK_LE(bitSum, WQbits);
    WQNdata[f] = 2 * bitSum - WQbits;
  }
}

void filterNormalizationL1(const TensorCPU& W, TensorCPU* WL1) {
  const auto F = W.dim32(0);
  WL1->Resize(F);
  const auto Ws = W.size() / F;
  const auto* Wdata = W.data<float>();
  auto* WL1data = WL1->mutable_data<float>();
  for (auto f = 0; f < F; ++f) {
    double l1sum = 0.0;
    for (auto j = 0; j < Ws; ++j) {
      l1sum += std::abs(Wdata[f * Ws + j]);
    }
    WL1data[f] = l1sum / Ws;
  }
}

void qim2col(const ConvArgs& args, const TensorCPU& XQ, const TensorCPU& WQ, TensorCPU* XQcol) {
  // TODO: pass pre-resized output?
  // TODO: handle strides?

  CAFFE_ENFORCE_EQ(XQ.dim32(3), WQ.dim32(3));
  const size_t N = XQ.dim32(0);
  const size_t IH = XQ.dim32(1);
  const size_t IW = XQ.dim32(2);
  const size_t KH = WQ.dim32(1);
  const size_t KW = WQ.dim32(2);
  const size_t KC = WQ.dim32(3);

  XQcol->Resize(XQ.dim32(0),
                (XQ.dim32(1) - KH + args.pad_t + args.pad_b) / args.stride_h + 1,
                (XQ.dim32(2) - KW + args.pad_l + args.pad_r) / args.stride_w + 1,
                KH * KW * KC);

  if (args.pad_l == 0 && args.pad_r == 0 && args.pad_b == 0 && args.pad_t == 0 &&
      args.stride_h == 1 && args.stride_w == 1 && KH == 1 && KW == 1) {
    CAFFE_ENFORCE_EQ(XQ.size(), XQcol->size());
    XQcol->ShareExternalPointer(const_cast<uint8_t*>(XQ.data<uint8_t>()), XQ.size());
    return;
  }
  const size_t OH = XQcol->dim32(1);
  const size_t OW = XQcol->dim32(2);

  const uint8_t* XQdata = XQ.data<uint8_t>();
  uint8_t* XQcoldata = XQcol->mutable_data<uint8_t>();
  for (size_t n = 0; n < N; ++n) {
    for (size_t oh = 0; oh < OH; ++oh) {
      int32_t h_pad = (int32_t)(args.stride_h * oh) - (int32_t)args.pad_t;
      for (size_t ow = 0; ow < OW; ++ow) {
        int32_t w_pad = (int32_t)(args.stride_w * ow) - (int32_t)args.pad_l;
        for (size_t kh = 0; kh < KH; ++kh) {
          int32_t ih = (int32_t)kh + h_pad;
          if ((size_t)ih < (size_t)IH && (size_t)w_pad < (size_t)IW &&
              (size_t)((int32_t)w_pad + (int32_t)KW) < (size_t)IW) {
            // We can do a larger memcpy, of size KW * KC
            size_t off = kh * KW * KC + ow * KH * KW * KC + oh * KH * KW * KC * OW +
                         n * KH * KW * KC * OW * OH;
            std::memcpy(&XQcoldata[off],
                        &XQdata[((int32_t)w_pad) * KC + ih * IW * KC + n * IW * KC * IH],
                        KW * KC);
          } else {
            for (size_t kw = 0; kw < KW; ++kw) {
              int32_t iw = (int32_t)kw + w_pad;
              // Use unsigned integer math to avoid multiple comparisons (>= H, < 0).
              size_t off = kw * KC + kh * KW * KC + ow * KH * KW * KC + oh * KH * KW * KC * OW +
                           n * KH * KW * KC * OW * OH;
              if ((size_t)ih < (size_t)IH && (size_t)iw < (size_t)IW) {
                std::memcpy(
                    &XQcoldata[off], &XQdata[iw * KC + ih * IW * KC + n * KC * IW * IH], KC);
              } else {
                // This should be simply padded with zero.
                std::memset(&XQcoldata[off], 0, KC);
              }
            }
          }
        }
      }
    }
  }
}

std::unique_ptr<QConvState> create2b1bConvState(Workspace* ws,
                                                const TensorCPU& W,
                                                const TensorCPU* b) {
  auto state = caffe2::make_unique<QConvState>();
  state->XQs.resize(k2b1bXBits);
  state->YQs.resize(k2b1bXBits);
  for (auto i = 0; i < k2b1bXBits; ++i) {
    state->XQs[i] = caffe2::make_unique<Tensor>(CPU);
    state->YQs[i] = caffe2::make_unique<Tensor>(CPU);
  }
  state->WQ = caffe2::make_unique<Tensor>(CPU);
  state->WQN = caffe2::make_unique<Tensor>(CPU);
  state->WQL1Norm = caffe2::make_unique<Tensor>(CPU);
  state->scratch = caffe2::make_unique<Tensor>(CPU);
  state->scratchColBuffer = caffe2::make_unique<Tensor>(CPU);

  signQuantize(W, state->WQ.get());
  filterNormalization11(*(state->WQ), state->WQN.get());
  filterNormalizationL1(W, state->WQL1Norm.get());
  // TODO: incorporate center distance normalization.
  // Since inputs to convs are [0, 1, 2, 3], instead of [0, x, 2 * x, ...],
  // we can just uniformly rescale the outputs by x, i.e.,
  // for (auto i = 0; i < r->WQL1Norm.size(); ++i) {
  //   r->WQL1Norm.mutable_data<float>()[i] *= center_distance;
  // }
  state->parallelFor = [ws](size_t range, std::function<void(size_t)> f) {
#if CAFFE2_MOBILE
    ws->GetThreadPool()->run([&](int, size_t v) { f(v); }, range);
#else
    for (size_t v = 0; v < range; ++v) {
      f(v);
    }
#endif
  };
  if (b) {
    state->bias = caffe2::make_unique<Tensor>(*b, CPU);
  }
  return state;
}

void run2b1bConvGeneric(QConvState* state, const ConvArgs& args, const TensorCPU& X, TensorCPU* Y) {
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  if (run2b1bConvNeon(state, args, X, Y)) {
    return;
  }
#endif
  uniformQuantize2b1b(X, state->XQs, 0.5, 1.0);
  for (auto i = 0; i < k2b1bXBits; ++i) {
    qconv(args, *(state->XQs[i]), *(state->WQ), nullptr, state->YQs[i].get());
  }
  Y->ResizeLike(*(state->YQs[0]));
  const auto F = state->WQ->dim(0);
  const auto N = Y->size() / F;
  run2b1bUnification(state,
                     N,
                     F,
                     state->WQN->data<float>(),
                     state->YQs[0]->data<float>(),
                     state->YQs[1]->data<float>(),
                     F,
                     Y->mutable_data<float>(),
                     F,
                     state->bias ? state->bias->data<float>() : nullptr);
}

void run2b1bUnification(QConvState* state,
                        size_t N,
                        size_t C,
                        const float* WQNVdata,
                        const float* YQs0Vdata,
                        const float* YQs1Vdata,
                        size_t YQstride,
                        float* Ydata,
                        size_t Ystride,
                        const float* bias) {
  ConstEigenVectorArrayMap<float> WQNV(WQNVdata, C);

  for (size_t j = 0; j < N; ++j) {
    ConstEigenVectorArrayMap<float> YQs0V(YQs0Vdata + YQstride * j, C);
    ConstEigenVectorArrayMap<float> YQs1V(YQs1Vdata + YQstride * j, C);
    EigenVectorArrayMap<float> YNV(Ydata + Ystride * j, C);
    if (bias) {
      ConstEigenVectorArrayMap<float> BV(bias, C);
      YNV = (std::pow<float>(2, k2b1bXBits) - 1) / 2 * WQNV + std::pow<float>(2, -1) * YQs0V +
            std::pow<float>(2, 0) * YQs1V + BV;
    } else {
      YNV = (std::pow<float>(2, k2b1bXBits) - 1) / 2 * WQNV + std::pow<float>(2, -1) * YQs0V +
            std::pow<float>(2, 0) * YQs1V;
    }
  }
}

class QConvOp final : public ConvPoolOpBase<CPUContext> {
 public:
  QConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws), ws_(ws) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NHWC, "QConvOp only supports NHWC order");
    OPERATOR_NEEDS_FEATURE(this->dilation_h() == 1, "");
    OPERATOR_NEEDS_FEATURE(this->dilation_w() == 1, "");
    OPERATOR_NEEDS_FEATURE(this->group_ == 1, "");
  }

  bool RunOnDeviceWithOrderNHWC() override {
    auto& X = Input(0);
    auto& filter = Input(1);
    const auto* bias = InputSize() == 3 ? &Input(2) : nullptr;
    auto* Y = Output(0);

    // TODO: Support multiple quantization methods instead of assuming 2b1b.
    if (!state_) {
      state_ = create2b1bConvState(ws_, filter, bias);
    }
    ConvArgs args;
    args.pad_l = this->pad_l();
    args.pad_t = this->pad_t();
    args.pad_b = this->pad_b();
    args.pad_r = this->pad_r();
    args.stride_h = this->stride_h();
    args.stride_w = this->stride_w();
    run2b1bConvGeneric(state_.get(), args, X, Y);
    return true;
  }

 private:
  std::unique_ptr<QConvState> state_;
  Workspace* ws_;
};

REGISTER_CPU_OPERATOR(QConv, QConvOp);

} // namespace caffe2
