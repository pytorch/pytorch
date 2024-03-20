#include "ulp_neon.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// TODO: tune this with cache size detection code. Changing to 32 helps on some
// devices (Snapdragon 820).
constexpr size_t kL1CacheSizeBytes = 16 * 1024;

#if defined(__ARM_NEON__) || defined(__ARM_NEON)

// Applies 2-bit uniform quantization to the floating point data at Xdata,
// storing QC bytes into XQdata (i.e. reading 8 * QC floats from Xdata).
// Requires QC to be a multiple of 8.
inline void quantize2bNeon(size_t QC,
                           const float* __restrict__ Xdata,
                           float offset,
                           float inter_center_distance,
                           std::array<uint8_t*, k2b1bXBits> XQdata) {
  TORCH_DCHECK_EQ(QC % 8, 0);
  const auto offset_plus_2_inter_center_distance = vdupq_n_f32(offset + 2 * inter_center_distance);
  const auto offset_plus_inter_center_distance = vdupq_n_f32(offset + inter_center_distance);
  const auto offset_ = vdupq_n_f32(offset);
  const uint8x8_t shifts = {1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1 << 7};

  for (size_t qc = 0; qc < QC; qc += 8) {
    std::array<std::array<uint8x8_t, 8>, k2b1bXBits> ps;
    for (auto i = 0; i < k2b1bXBits; ++i) {
      for (auto j = 0; j < 8; ++j) {
        ps[i][j] = vdup_n_u8(0);
      }
    }

    for (auto j = 0; j < 8; ++j) {
      const auto x0 = vld1q_f32(&Xdata[qc * 8 + j * 8 + 0]);
      const auto x1 = vld1q_f32(&Xdata[qc * 8 + j * 8 + 4]);

      // logic.
      // if (v >= offset + inter_center_distance) {
      //   p[1] |= 1 << b;
      // } else {
      //   p[1] |= 0 << b;
      // }

      // if ((v >= offset && v < offset + inter_center_distance) ||
      //     (v >= offset * 2 * inter_center_distance)) {
      //   p[0] |= 1 << b;
      // } else {
      //   p[0] |= 0 << b;
      // }

      auto join = [](uint32x4_t a, uint32x4_t b) -> uint8x8_t {
        return vmovn_u16(vcombine_u16(vmovn_u32(a), vmovn_u32(b)));
      };

      const auto x_geq_offset_plus_2_inter_center_distance =
          join(vcgeq_s32(vreinterpretq_s32_f32(x0),
                         vreinterpretq_s32_f32(offset_plus_2_inter_center_distance)),
               vcgeq_s32(vreinterpretq_s32_f32(x1),
                         vreinterpretq_s32_f32(offset_plus_2_inter_center_distance)));
      const auto x_ge_offset =
          join(vcgeq_s32(vreinterpretq_s32_f32(x0), vreinterpretq_s32_f32(offset_)),
               vcgeq_s32(vreinterpretq_s32_f32(x1), vreinterpretq_s32_f32(offset_)));

      const auto x_lt_offset_plus_inter_center_distance =
          join(vcltq_s32(vreinterpretq_s32_f32(x0),
                         vreinterpretq_s32_f32(offset_plus_inter_center_distance)),
               vcltq_s32(vreinterpretq_s32_f32(x1),
                         vreinterpretq_s32_f32(offset_plus_inter_center_distance)));

      const auto p1_mask = vmvn_u8(x_lt_offset_plus_inter_center_distance);
      const auto p0_mask = vorr_u8(vand_u8(x_ge_offset, x_lt_offset_plus_inter_center_distance),
                                   x_geq_offset_plus_2_inter_center_distance);
      ps[0][j] = vand_u8(shifts, p0_mask);
      ps[1][j] = vand_u8(shifts, p1_mask);
    }

    for (auto i = 0; i < 2; ++i) {
      const auto p01 = vpadd_u8(ps[i][0], ps[i][1]);
      const auto p23 = vpadd_u8(ps[i][2], ps[i][3]);
      const auto p45 = vpadd_u8(ps[i][4], ps[i][5]);
      const auto p67 = vpadd_u8(ps[i][6], ps[i][7]);
      const auto p0123 = vpadd_u8(p01, p23);
      const auto p4567 = vpadd_u8(p45, p67);
      vst1_u8(XQdata[i] + qc, vpadd_u8(p0123, p4567));
    }
  }
}

void uniformQuantize2b1bNeon(QConvState* state,
                             const TensorCPU& X,
                             const std::vector<std::unique_ptr<TensorCPU>>& XQ,
                             float offset,
                             float inter_center_distance) {
  CAFFE_ENFORCE_GT(X.ndim(), 1);
  const size_t C = X.dim32(X.ndim() - 1);
  const size_t N = X.size() / C;
  const size_t QC = divRoundUp(C, 8);
  auto XQs = X.sizes().vec();
  XQs[X.ndim() - 1] = QC;
  CAFFE_ENFORCE_EQ(XQ.size(), k2b1bXBits);
  for (auto i = 0; i < k2b1bXBits; ++i) {
    XQ[i]->Resize(XQs);
  }
  const float* Xdata = X.data<float>();
  std::array<uint8_t*, k2b1bXBits> XQdata;
  for (size_t i = 0; i < k2b1bXBits; ++i) {
    XQdata[i] = XQ[i]->mutable_data<uint8_t>();
  }
  CAFFE_ENFORCE_GT(offset, 0);
  CAFFE_ENFORCE_GT(inter_center_distance, 0);
  size_t QCUnroll = ((C / 8) / 8) * 8;
  // Each worker loads an L1 cache sized block.
  // We read/write B * K * 4 + 2 * B * (K / 8), so to fit inside C, we have
  // B = 4 * C / 17 K.
  // QCUnroll = 0;
  const size_t rowsPerBlock =
      std::max<size_t>(std::floor<size_t>(double(4 * kL1CacheSizeBytes) / double(17 * C)), 1);
  state->parallelFor(divRoundUp(N, rowsPerBlock), [&](size_t nb) {
    for (size_t n = nb * rowsPerBlock; n < std::min<size_t>(nb * rowsPerBlock + rowsPerBlock, N);
         ++n) {
      std::array<uint8_t*, k2b1bXBits> XQoff = {{
          XQdata[0] + 0 + QC * n, XQdata[1] + 0 + QC * n,
      }};
      quantize2bNeon(QCUnroll, &Xdata[0 + C * n], offset, inter_center_distance, XQoff);
      for (size_t qc = QCUnroll; qc < QC; ++qc) {
        // compute the block in X.
        std::array<uint8_t, k2b1bXBits> p = {{0, 0}};
        for (size_t b = 0; b < 8; ++b) {
          const size_t c = qc * 8 + b;
          if (c < C) {
            float v = Xdata[c + C * n];
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
  });
}

template <size_t TileSize, size_t TileDepthBytes>
void uniformQuantize2b1bNeonPacked(QConvState* state,
                                   const TensorCPU& X,
                                   const std::vector<std::unique_ptr<TensorCPU>>& XQ,
                                   float offset,
                                   float inter_center_distance) {
  const size_t M = X.size_to_dim(3);
  const size_t K = X.size() / M;
  const size_t QK = divRoundUp(K, 8);
  const size_t numTiles = divRoundUp(M, TileSize);
  const size_t numTilesDepth = divRoundUp(QK, TileDepthBytes);
  for (size_t i = 0; i < k2b1bXBits; ++i) {
    XQ[i]->Resize(numTiles, numTilesDepth, TileSize, TileDepthBytes);
  }
  const float* Xdata = X.data<float>();
  std::array<uint8_t*, k2b1bXBits> XQdata;
  for (auto i = 0; i < k2b1bXBits; ++i) {
    XQdata[i] = XQ[i]->mutable_data<uint8_t>();
  }
  CAFFE_ENFORCE_GT(offset, 0);
  CAFFE_ENFORCE_GT(inter_center_distance, 0);
  // Each worker loads an L1 cache sized block.
  // We read/write B * K * TileSize * 4 + 2 * B * TileSize * (K / 8), so to fit inside C, we have
  // B = 4 * C / (17 * K * TileSize).
  const size_t tilesPerBlock = std::max<size_t>(
      std::floor<size_t>(double(4 * kL1CacheSizeBytes) / double(17 * K * TileSize)), 1);
  state->parallelFor(divRoundUp(numTiles, tilesPerBlock), [&](size_t nb) {
    for (size_t i = nb * tilesPerBlock;
         i < std::min<size_t>(nb * tilesPerBlock + tilesPerBlock, numTiles);
         ++i) {
      for (size_t j = 0; j < numTilesDepth; ++j) {
        if (i != numTiles - 1 && j != numTilesDepth - 1) {
          // we have a full tile. Just memcpy.
          for (auto ii = 0; ii < TileSize; ++ii) {
            size_t m = i * TileSize + ii;
            size_t k = j * TileDepthBytes * 8;
            std::array<uint8_t*, k2b1bXBits> XQoff = {
                {XQdata[0] + TileDepthBytes * ii + TileDepthBytes * TileSize * j +
                     TileSize * TileDepthBytes * numTilesDepth * i,
                 XQdata[1] + TileDepthBytes * ii + TileDepthBytes * TileSize * j +
                     TileSize * TileDepthBytes * numTilesDepth * i}};
            quantize2bNeon(TileDepthBytes, &Xdata[m * K + k], offset, inter_center_distance, XQoff);
          }
        } else {
          for (size_t ii = 0; ii < TileSize; ++ii) {
            size_t m = i * TileSize + ii;
            size_t k = j * TileDepthBytes * 8;
            std::array<uint8_t*, k2b1bXBits> XQoff = {
                {XQdata[0] + TileDepthBytes * ii + TileDepthBytes * TileSize * j +
                     TileSize * TileDepthBytes * numTilesDepth * i,
                 XQdata[1] + TileDepthBytes * ii + TileDepthBytes * TileSize * j +
                     TileSize * TileDepthBytes * numTilesDepth * i}};
            if (m < M && k + TileDepthBytes * 8 <= K) {
              // We can just read the stripe directly.
              quantize2bNeon(
                  TileDepthBytes, &Xdata[m * K + k], offset, inter_center_distance, XQoff);
            } else {
              // We need to pad the stripe to the full amount read by
              // quantize2bNeon.
              std::array<float, 8 * TileDepthBytes> Xpad = {{0}};
              if (m < M) {
                std::copy(&Xdata[m * K + k], &Xdata[m * K + K], Xpad.begin());
              }
              quantize2bNeon(TileDepthBytes, Xpad.data(), offset, inter_center_distance, XQoff);
            }
          }
        }
      }
    }
  });
}

// Packs a matrix (of size MxK) into a tiled array of size
// (M/TileSize)x(K/TileDepthBytes)xTileSizexTileDepthBytes.
template <size_t TileSize, size_t TileDepthBytes>
void qpack_tiles(QConvState* state, const TensorCPU& X, size_t axis, TensorCPU* XP) {
  const size_t M = X.size_to_dim(axis);
  const size_t QK = X.size() / M;
  const size_t numTiles = divRoundUp(M, TileSize);
  const size_t numTilesDepth = divRoundUp(QK, TileDepthBytes);
  XP->Resize(numTiles, numTilesDepth, TileSize, TileDepthBytes);

  const auto* __restrict__ Xdata = X.data<uint8_t>();
  auto* __restrict__ XPdata = XP->mutable_data<uint8_t>();
  // Load L1 sized tiles per thread.
  // We read/write 2 * B * QK * TileSize bytes, so
  // B = C / (2 * QK * TileSize)
  const size_t tilesPerBlock = std::max<size_t>(
      std::floor<size_t>(double(kL1CacheSizeBytes) / double(2 * TileSize * QK)), 1);
  state->parallelFor(divRoundUp(numTiles, tilesPerBlock), [&](size_t nb) {
    for (size_t i = nb * tilesPerBlock;
         i < std::min<size_t>(nb * tilesPerBlock + tilesPerBlock, numTiles);
         ++i) {
      for (size_t j = 0; j < numTilesDepth; ++j) {
        if (i != numTiles - 1 && j != numTilesDepth - 1) {
          // we have a full tile. Just memcpy.
          for (auto ii = 0; ii < TileSize; ++ii) {
            auto m = i * TileSize + ii;
            auto qk = j * TileDepthBytes;
            std::memcpy(&XPdata[TileDepthBytes * ii + TileDepthBytes * TileSize * j +
                                TileSize * TileDepthBytes * numTilesDepth * i],
                        &Xdata[m * QK + qk],
                        TileDepthBytes);
          }
        } else {
          for (size_t ii = 0; ii < TileSize; ++ii) {
            for (size_t jj = 0; jj < TileDepthBytes; ++jj) {
              size_t m = i * TileSize + ii;
              size_t qk = j * TileDepthBytes + jj;
              uint8_t pval = 0;
              if (m < M && qk < QK) {
                // get value from X
                pval = Xdata[m * QK + qk];
              }
              XPdata[jj + TileDepthBytes * ii + TileDepthBytes * TileSize * j +
                     TileSize * TileDepthBytes * numTilesDepth * i] = pval;
            }
          }
        }
      }
    }
  });
}

// Computes the kUnrollM x kUnrollM tile of a GEMM by multiplying two packed
// slices of size (kUnrolLMxK). These tiles are constructed by the qpack_tiles
// function, which packs an input array of size [M][K] into an
// [M/TileSize][K/TileDepthBytes][TileSize][TileDepthBytes], which ensures all
// the array accesses in this function is contiguous.
template <size_t kUnrollM, size_t kUnrollN, size_t TileDepthBytes, typename F>
void qgess_packed(const uint8_t* __restrict__ Ablock,
                  const uint8_t* __restrict__ Bblock,
                  float* __restrict__ Cblock,
                  const size_t Cstride,
                  const size_t QK,
                  const size_t Nstart,
                  F&& f) {
  static_assert(kUnrollN % 8 == 0, "");
  static_assert(TileDepthBytes == 16, "");
  TORCH_DCHECK_EQ(QK % 16, 0);
  uint16x8_t acc[kUnrollM][kUnrollN / 8];
  for (size_t mm = 0; mm < kUnrollM; ++mm) {
    for (size_t nn = 0; nn < kUnrollN / 8; ++nn) {
      acc[mm][nn] = vdupq_n_u16(0);
    }
  }
  size_t qk = 0;
  const size_t QK16Unroll = (QK / 16) * 16;
  for (; qk < QK16Unroll; qk += 16) {
    uint8x16_t Areg[kUnrollM];
    for (size_t mm = 0; mm < kUnrollM; ++mm) {
      Areg[mm] = vld1q_u8(Ablock);
      Ablock += 16;
    }

    for (size_t nn = 0; nn < kUnrollN / 8; ++nn) {
      uint8x16_t Breg[8];
      for (size_t nnn = 0; nnn < 8; ++nnn) {
        Breg[nnn] = vld1q_u8(Bblock);
        Bblock += 16;
      }
      for (size_t mm = 0; mm < kUnrollM; ++mm) {
        uint8x16_t cnts[8];
        for (size_t nnn = 0; nnn < 8; ++nnn) {
          cnts[nnn] = vcntq_u8(veorq_u8(Breg[nnn], Areg[mm]));
        }
        uint8x8_t ps[8];
        for (size_t nnn = 0; nnn < 8; ++nnn) {
          ps[nnn] = vadd_u8(vget_low_u8(cnts[nnn]), vget_high_u8(cnts[nnn]));
        }
        uint8x8_t pss[4];
        for (size_t nnn = 0; nnn < 4; ++nnn) {
          pss[nnn] = vpadd_u8(ps[2 * nnn], ps[2 * nnn + 1]);
        }
        uint8x8_t psss[2];
        for (size_t nnn = 0; nnn < 2; ++nnn) {
          psss[nnn] = vpadd_u8(pss[2 * nnn], pss[2 * nnn + 1]);
        }
        uint8x16_t out = vcombine_u8(psss[0], psss[1]);
        acc[mm][nn] = vpadalq_u8(acc[mm][nn], out);
      }
    }
  }

  for (size_t mm = 0; mm < kUnrollM; ++mm) {
    auto* Crow = Cblock + mm * Cstride;
    for (size_t nn = 0; nn < kUnrollN / 8; ++nn) {
      const int32x4_t K_ = vdupq_n_s32(QK * 8);
      const int16x4_t two = vdup_n_s16(2);
      const int16x4_t acc0123_l = vreinterpret_s16_u16(vget_low_u16(acc[mm][nn]));
      const int16x4_t acc0123_h = vreinterpret_s16_u16(vget_high_u16(acc[mm][nn]));
      const int32x4_t K_minus_2_acc0123_l = vmlsl_s16(K_, two, acc0123_l);
      const int32x4_t K_minus_2_acc0123_h = vmlsl_s16(K_, two, acc0123_h);
      f(Crow + nn * 8 + 0, vcvtq_f32_s32(K_minus_2_acc0123_l), Nstart + nn * 8 + 0);
      f(Crow + nn * 8 + 4, vcvtq_f32_s32(K_minus_2_acc0123_h), Nstart + nn * 8 + 4);
    }
  }
}

// Computes the (normal + transpose) matrix-matrix product of two -1/1 binary
// matrices, laid out in the standard format.
template <size_t TileSize, size_t TileDepthBytes, typename F>
inline void qgemm_nt_packed(
    QConvState* state, const TensorCPU& A, const TensorCPU& B, TensorCPU* C, F&& f = F()) {
  CAFFE_ENFORCE_EQ(A.ndim(), 4);
  CAFFE_ENFORCE_EQ(B.ndim(), 4);
  CAFFE_ENFORCE_EQ(A.dim(2), TileSize);
  CAFFE_ENFORCE_EQ(B.dim(2), TileSize);
  CAFFE_ENFORCE_EQ(A.dim(3), TileDepthBytes);
  CAFFE_ENFORCE_EQ(B.dim(3), TileDepthBytes);
  const size_t MT = A.dim(0);
  const size_t NT = B.dim(0);
  const size_t M = MT * TileSize;
  const size_t N = NT * TileSize;

  const size_t QKT = A.dim(1);
  const size_t K = QKT * 8 * TileDepthBytes;
  const size_t QK = K / 8;
  CAFFE_ENFORCE_EQ(A.dim(1), B.dim(1));
  C->Resize(M, N);
  const auto* Adata = A.data<uint8_t>();
  const auto* Bdata = B.data<uint8_t>();
  auto* Cdata = C->mutable_data<float>();

  // Assume TxT tile. Each input slice is of size T x (K/8) bytes, and the output
  // is a tile of size T x T x sizeof(float) bytes. We want the sum of this to fit
  // in L1 cache. This means for a block number of tiles B , we load B * T * K /
  // 8 + B * T * K / 8 + B * B * T * T * sizeof(float).

  // If cache size = C, we get
  // B = 1/(32 * T) (sqrt(256 C + K^2) - K)
  // taking floor (by integer division), gives the result.

  // Assume 16KB L1 cache.
  size_t tilesPerBlock =
      std::floor((std::sqrt(256 * kL1CacheSizeBytes + K * K) - K) / (32 * TileSize));
  if (tilesPerBlock < 1) {
    tilesPerBlock = 1;
  }
  CAFFE_ENFORCE_LT(K, std::pow(2, 16));
  CAFFE_ENFORCE_EQ(M % TileSize, 0);
  CAFFE_ENFORCE_EQ(N % TileSize, 0);
  const size_t MNumTiles = M / TileSize;
  const size_t NNumTiles = N / TileSize;
  const size_t MNumBlocks = divRoundUp(MNumTiles, tilesPerBlock);
  const size_t NNumBlocks = divRoundUp(NNumTiles, tilesPerBlock);

  state->parallelFor(MNumBlocks * NNumBlocks, [&](size_t mn) {
    const size_t mBlockIdx = mn / NNumBlocks;
    const size_t nBlockIdx = mn % NNumBlocks;
    const size_t mTileStart = mBlockIdx * tilesPerBlock;
    const size_t nTileStart = nBlockIdx * tilesPerBlock;
    for (size_t mBlockTileIdx = 0;
         mBlockTileIdx < tilesPerBlock && mBlockTileIdx + mTileStart < MNumTiles;
         ++mBlockTileIdx) {
      const size_t mTileIdx = mBlockTileIdx + mTileStart;
      for (size_t nBlockTileIdx = 0;
           nBlockTileIdx < tilesPerBlock && nBlockTileIdx + nTileStart < NNumTiles;
           ++nBlockTileIdx) {
        const size_t nTileIdx = nBlockTileIdx + nTileStart;
        // A layout: [M/TileSize][QK / TileDepth][TileSize][TileDepth]
        // C layout: [M/TileSize][TileSize][N/TileSize][TileSize]
        const auto* Ablock = &Adata[mTileIdx * QK * TileSize];
        const auto* Bblock = &Bdata[nTileIdx * QK * TileSize];
        auto* Cblock = &Cdata[mTileIdx * TileSize * N + nTileIdx * TileSize];
        const size_t Cstride = N;
        qgess_packed<TileSize, TileSize, TileDepthBytes, F>(
            Ablock, Bblock, Cblock, Cstride, QK, nTileIdx * TileSize, std::forward<F>(f));
      }
    }
  });
}

void run2b1bConvIm2ColGEMM(QConvState* state,
                           const ConvArgs& args,
                           const TensorCPU& X,
                           TensorCPU* Y) {
  // TODO: packing + quantization in same block.
  const size_t KH = state->WQ->dim32(1);
  const size_t KW = state->WQ->dim32(2);
  const size_t OH = (X.dim32(1) - KH + args.pad_t + args.pad_b) / args.stride_h + 1;
  const size_t OW = (X.dim32(2) - KW + args.pad_l + args.pad_r) / args.stride_w + 1;
  const size_t OC = state->WQ->dim32(0);
  const size_t QK = KH * KW * divRoundUp(X.dim32(3), 8);
  Y->Resize(X.dim32(0), OH, OW, OC);
  if (!state->WQPacked) {
    state->WQPacked = std::make_unique<Tensor>(CPU);
    qpack_tiles<kGEMMTileSize, kGEMMTileDepthBytes>(state, *(state->WQ), 1, state->WQPacked.get());
    CAFFE_ENFORCE_EQ(state->WQPacked->dim32(0), divRoundUp(OC, kGEMMTileSize));
    CAFFE_ENFORCE_EQ(state->WQPacked->dim32(1), divRoundUp(QK, kGEMMTileDepthBytes));
    CAFFE_ENFORCE_EQ(state->WQPacked->dim32(2), kGEMMTileSize);
    CAFFE_ENFORCE_EQ(state->WQPacked->dim32(3), kGEMMTileDepthBytes);

    // We can fuse the bias addition into the filter normalization. We can
    // replace the bias + 3/2 normalization factor by replacing normalization
    // with (2/3 bias + normalization), and setting bias to zero.
    if (state->bias) {
      for (size_t i = 0; i < state->bias->size(); ++i) {
        state->WQN->mutable_data<float>()[i] += 2.0f / 3 * state->bias->data<float>()[i];
      }
    }
    state->bias.reset();

    // If we have to pad when we pack our weight tiles, then we need to adjust
    // the normalization factor by the number of zeros that we added.
    const size_t QKPadding = divRoundUp(QK, kGEMMTileDepthBytes) * kGEMMTileDepthBytes - QK;
    if (QKPadding != 0) {
      for (size_t i = 0; i < state->WQN->size(); ++i) {
        state->WQN->mutable_data<float>()[i] -= QKPadding * 8;
      }
    }
  }
  CAFFE_ENFORCE(!state->bias.get());
  // Since 1x1s are so common, we fuse the quantization + packing steps.
  const bool is_1x1 = KH == 1 && KW == 1 && args.pad_l == 0 && args.pad_r == 0 && args.pad_b == 0 &&
                      args.pad_t == 0 && args.stride_h == 1 && args.stride_w == 1;

  if (is_1x1) {
    CAFFE_ENFORCE_EQ(OH, X.dim32(1));
    CAFFE_ENFORCE_EQ(OW, X.dim32(2));
    uniformQuantize2b1bNeonPacked<kGEMMTileSize, kGEMMTileDepthBytes>(
        state, X, state->XQs, 0.5, 1.0);
  } else {
    uniformQuantize2b1bNeon(state, X, state->XQs, 0.5, 1.0);
  }
  TensorCPU* YQ0 = state->YQs[0].get();

  if (state->WQ->dim32(0) % kGEMMTileSize == 0) {
    // We can run inplace by operating on our Y vector, and then shrinking Y.
    YQ0 = Y;
  }

  for (size_t i = 0; i < k2b1bXBits; ++i) {
    const auto& XQ = *(state->XQs[i]);
    if (!is_1x1) {
      qim2col(args, XQ, *(state->WQ), state->scratchColBuffer.get());
      qpack_tiles<kGEMMTileSize, kGEMMTileDepthBytes>(
          state, *(state->scratchColBuffer), 3, state->scratch.get());
    }

    {
      const auto* __restrict__ WQNdata = state->WQN->data<float>();
      switch (i) {
      case 0:
        qgemm_nt_packed<kGEMMTileSize, kGEMMTileDepthBytes>(
            state,
            is_1x1 ? XQ : *(state->scratch),
            *(state->WQPacked),
            YQ0,
            [WQNdata](float* __restrict__ acc, float32x4_t value, size_t channel) {
              // acc[c] = 3/2 WQN[c] + 1/2 value[c];
              const float32x4_t _32 = vdupq_n_f32(3.0f / 2);
              const float32x4_t _12 = vdupq_n_f32(1.0f / 2);
              const float32x4_t WQNc_32 = vmulq_f32(_32, vld1q_f32(WQNdata + channel));
              const float32x4_t WQNc_32_value_12 = vmlaq_f32(WQNc_32, _12, value);
              vst1q_f32(acc, WQNc_32_value_12);
            });
        break;
      case 1:
        qgemm_nt_packed<kGEMMTileSize, kGEMMTileDepthBytes>(
            state,
            is_1x1 ? XQ : *(state->scratch),
            *(state->WQPacked),
            YQ0,
            [](float* __restrict__ acc, float32x4_t value, size_t channel) {
              const float32x4_t curr = vld1q_f32(acc);
              vst1q_f32(acc, vaddq_f32(curr, value));
            });
        break;
      }
    }
  }

  if (YQ0 != Y) {
    // In this case, the stride does not match, so we need to copy the output
    // data into the contiguous Y matrix.
    const size_t F = state->WQ->dim(0);
    const size_t N = Y->size() / F;
    const size_t NP = YQ0->dim32(0);
    const size_t FP = YQ0->dim32(1);
    math::CopyMatrix<CPUContext>(
        sizeof(float), N, F, YQ0->data<float>(), FP, Y->mutable_data<float>(), F, nullptr);
  } else {
    CAFFE_ENFORCE_EQ(Y->dim32(0), divRoundUp(X.dim32(0) * OH * OW, kGEMMTileSize) * kGEMMTileSize);
    CAFFE_ENFORCE_EQ(Y->dim32(1), OC);
    Y->ShrinkTo(X.dim32(0) * OH * OW);
    Y->Reshape(std::vector<int64_t>{{int64_t(X.dim(0)), int64_t(OH), int64_t(OW), int64_t(OC)}});
  }
}

bool run2b1bConvNeon(QConvState* state, const ConvArgs& args, const TensorCPU& X, TensorCPU* Y) {
  // TODO: insert specialized cases (e.g. depthwise convolutions, the direct
  // convolution.
  CAFFE_ENFORCE_EQ(X.ndim(), 4);
  run2b1bConvIm2ColGEMM(state, args, X, Y);
  return true;
}

#endif

} // namespace caffe2
