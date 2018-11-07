#ifndef CAFFE2_OPERATORS_INT8_CHANNEL_SHUFFLE_OP_H_
#define CAFFE2_OPERATORS_INT8_CHANNEL_SHUFFLE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/quantized/int8_simd.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

namespace int8 {

namespace {

template <size_t TileSizeK, size_t TileSizeG>
inline void
TransposeTile(const uint8_t* X_tile, size_t K, size_t G, uint8_t* Y_tile) {
#ifdef INT8_NEON_SIMD
  static_assert(TileSizeK == 8, "");
  static_assert(TileSizeG == 4, "");
  auto Transpose8x4_NEON =
      [](uint8x8_t* a0, uint8x8_t* a1, uint8x8_t* a2, uint8x8_t* a3) {
        const uint8x8x2_t b0 = vtrn_u8(*a0, *a1);
        const uint8x8x2_t b1 = vtrn_u8(*a2, *a3);
        const uint16x4x2_t c0 = vtrn_u16(
            vreinterpret_u16_u8(b0.val[0]), vreinterpret_u16_u8(b1.val[0]));
        const uint16x4x2_t c1 = vtrn_u16(
            vreinterpret_u16_u8(b0.val[1]), vreinterpret_u16_u8(b1.val[1]));
        *a0 = vreinterpret_u8_u16(c0.val[0]);
        *a1 = vreinterpret_u8_u16(c1.val[0]);
        *a2 = vreinterpret_u8_u16(c0.val[1]);
        *a3 = vreinterpret_u8_u16(c1.val[1]);
      };

  uint8x8_t g0 = vld1_u8(X_tile + 0 * K);
  uint8x8_t g1 = vld1_u8(X_tile + 1 * K);
  uint8x8_t g2 = vld1_u8(X_tile + 2 * K);
  uint8x8_t g3 = vld1_u8(X_tile + 3 * K);
  Transpose8x4_NEON(&g0, &g1, &g2, &g3);
  uint8_t tile[TileSizeK / 2][2][TileSizeG];
  vst1_u8(&tile[0][0][0], g0);
  vst1_u8(&tile[1][0][0], g1);
  vst1_u8(&tile[2][0][0], g2);
  vst1_u8(&tile[3][0][0], g3);
  for (auto kkk = 0; kkk < 2; ++kkk) {
    for (auto kk = 0; kk < TileSizeK / 2; ++kk) {
      const auto k = TileSizeK / 2 * kkk + kk;
      for (auto g = 0; g < TileSizeG; ++g) {
        Y_tile[k * G + g] = tile[kk][kkk][g];
      }
    }
  }
#else
  uint8_t tile[TileSizeG][TileSizeK];
  for (auto g = 0; g < TileSizeG; ++g) {
    for (auto k = 0; k < TileSizeK; ++k) {
      tile[g][k] = X_tile[g * K + k];
    }
  }
  for (auto k = 0; k < TileSizeK; ++k) {
    for (auto g = 0; g < TileSizeG; ++g) {
      Y_tile[k * G + g] = tile[g][k];
    }
  }
#endif
}

void Int8ChannelShuffle(
    const uint8_t* X_data,
    size_t B,
    size_t K,
    size_t G,
    uint8_t* Y_data,
    ThreadPool* threadPool) {
  auto divRoundUp = [](size_t n, size_t d) { return (n + d - 1) / d; };
  constexpr size_t kTileSizeG = 4;
  constexpr size_t kTileSizeK = 8;
  auto f = [&](int, size_t b) {
    for (auto kk = 0; kk < divRoundUp(K, kTileSizeK); ++kk) {
      for (auto gg = 0; gg < divRoundUp(G, kTileSizeG); ++gg) {
        const auto g = gg * kTileSizeG;
        const auto k = kk * kTileSizeK;
        const auto X_tile = X_data + b * G * K + g * K + k;
        auto* Y_tile = Y_data + b * G * K + k * G + g;
        if (kk * kTileSizeK + kTileSizeK <= K &&
            gg * kTileSizeG + kTileSizeG <= G) {
          // Complete tile.
          TransposeTile<kTileSizeK, kTileSizeG>(X_tile, K, G, Y_tile);
        } else {
          uint8_t Xp[kTileSizeG][kTileSizeK];
          uint8_t Yp[kTileSizeK][kTileSizeG];
          for (auto kt = 0; kt < kTileSizeK; ++kt) {
            for (auto gt = 0; gt < kTileSizeG; ++gt) {
              if (k + kt < K && g + gt < G) {
                Xp[gt][kt] = X_tile[gt * K + kt];
              }
            }
          }
          TransposeTile<kTileSizeK, kTileSizeG>(
              &Xp[0][0], kTileSizeK, kTileSizeG, &Yp[0][0]);
          for (auto kt = 0; kt < kTileSizeK; ++kt) {
            for (auto gt = 0; gt < kTileSizeG; ++gt) {
              if (k + kt < K && g + gt < G) {
                Y_tile[kt * G + gt] = Yp[kt][gt];
              }
            }
          }
        }
      }
    }
  };
  threadPool->run(f, B);
}

} // namespace

class Int8ChannelShuffleOp final : public ConvPoolOpBase<CPUContext> {
 public:
  Int8ChannelShuffleOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws),
        ws_(ws) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NHWC,
        "Int8ChannelShuffleOp only supports NHWC order");
  }

  bool RunOnDeviceWithOrderNHWC() override {
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    Y->t.ResizeLike(X.t);
    Y->scale = X.scale;
    Y->zero_point = X.zero_point;
    int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
    auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    CHECK_EQ(Y_offset, X.zero_point);
    CHECK_EQ(Y_scale, X.scale);
    CHECK_GE(X.zero_point, std::numeric_limits<uint8_t>::min());
    CHECK_LE(X.zero_point, std::numeric_limits<uint8_t>::max());

    const auto C = X.t.dim32(3);
    CAFFE_ENFORCE(C % this->group_ == 0, "");
    const auto G = this->group_;
    const auto K = C / G;
    const auto B = X.t.dim32(0) * X.t.dim32(1) * X.t.dim32(2);
    Int8ChannelShuffle(
        X.t.data<uint8_t>(),
        B,
        K,
        G,
        Y->t.mutable_data<uint8_t>(),
        ws_->GetThreadPool());
    return true;
  }

 private:
  Workspace* ws_;
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_CHANNEL_SHUFFLE_OP_H_
