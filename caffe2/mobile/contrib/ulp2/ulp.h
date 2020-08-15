#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

constexpr size_t k2b1bXBits = 2;

struct ConvArgs {
  int stride_w{1};
  int stride_h{1};
  int pad_l{0};
  int pad_t{0};
  int pad_b{0};
  int pad_r{0};
};

using ParallelFor = std::function<void(size_t, std::function<void(size_t)>)>;

struct QConvState {
  std::vector<std::unique_ptr<TensorCPU>> XQs;
  std::vector<std::unique_ptr<TensorCPU>> YQs;
  std::unique_ptr<TensorCPU> WQ;
  // architecture-dependent whether packing is used.
  std::unique_ptr<TensorCPU> WQPacked;
  std::unique_ptr<TensorCPU> WQN;
  std::unique_ptr<TensorCPU> WQL1Norm;
  // Useful for e.g. incomplete tiles
  std::unique_ptr<TensorCPU> scratch;
  std::unique_ptr<TensorCPU> scratchColBuffer;

  std::unique_ptr<TensorCPU> bias;

  ParallelFor parallelFor{nullptr};
};

void uniformQuantize2b1b(const TensorCPU& X,
                         const std::vector<std::unique_ptr<TensorCPU>>& XQ,
                         float offset,
                         float inter_center_distance);

void qpad_zero(const ConvArgs& args, const TensorCPU& X, TensorCPU* Y);

inline size_t divRoundUp(size_t x, size_t d) { return (x + d - 1) / d; }

void signQuantize(const TensorCPU& X, TensorCPU* XQ);
void filterNormalization11(const TensorCPU& WQ, TensorCPU* WQN);
void filterNormalizationL1(const TensorCPU& W, TensorCPU* WL1);
std::unique_ptr<QConvState> create2b1bConvState(Workspace* ws,
                                                const TensorCPU& W,
                                                const TensorCPU* b);
void run2b1bConvGeneric(QConvState* state, const ConvArgs& args, const TensorCPU& X, TensorCPU* Y);
void qconv(
    const ConvArgs& args, const TensorCPU& X, const TensorCPU& W, const TensorCPU* b, TensorCPU* Y);
void qim2col(const ConvArgs& args, const TensorCPU& XQ, const TensorCPU& WQ, TensorCPU* XQcol);

void run2b1bUnification(QConvState* state,
                        size_t N,
                        size_t C,
                        const float* WQNVdata,
                        const float* YQs0Vdata,
                        const float* YQs1Vdata,
                        size_t YQstride,
                        float* Ydata,
                        size_t Ystride,
                        const float* bias);

} // namespace caffe2
