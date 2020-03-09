#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace {

void RowwiseMaxAndArg(
    const float* mat,
    int32_t N,
    int32_t D,
    float* rowMax,
    int32_t* argMax) {
  auto eigenMat = ConstEigenMatrixMap<float>(mat, D, N);
  for (auto i = 0; i < D; i++) {
    // eigenMat.row(i) is equivalent to column i in mat
    rowMax[i] = eigenMat.row(i).maxCoeff(argMax + i);
  }
}
void ColwiseMaxAndArg(
    const float* mat,
    int32_t N,
    int32_t D,
    float* colMax,
    int32_t* argMax) {
  auto eigenMat = ConstEigenMatrixMap<float>(mat, D, N);
  for (auto i = 0; i < N; i++) {
    // eigenMat.col(i) is equivalent to row i in mat
    colMax[i] = eigenMat.col(i).maxCoeff(argMax + i);
  }
}

class ViterbiPathOp : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit ViterbiPathOp(Args&&... args)
      : Operator(std::forward<Args>(args)...) {}

  void GatherRow(
      const TensorCPU& data,
      int32_t rowIndex,
      int32_t block_size,
      int32_t block_bytesize,
      TensorCPU* outRow) {
    CAFFE_ENFORCE(
        0 <= rowIndex && rowIndex < data.size(0),
        "rowIndex is out of DATA bounds");
    auto out = static_cast<char*>(outRow->raw_mutable_data(data.dtype()));
    auto src_base = static_cast<const char*>(data.raw_data());
    auto src = src_base + rowIndex * block_bytesize;
    context_.CopyItemsSameDevice(data.dtype(), block_size, src, out);
  }

  void
  AddColToMat(const TensorCPU& mat, const TensorCPU& col, TensorCPU* result) {
    float* resultData = result->template mutable_data<float>();
    const float* colData = col.template data<float>();
    // Initialize the columns of the result to be = the input col
    for (auto i = 0; i < result->dim32(1); i++) {
      for (auto j = 0; j < result->dim32(0); j++) {
        resultData[i * result->dim32(0) + j] = colData[i];
      }
    }
    // Element-wise add of the result and the input matrix
    math::Add<float, CPUContext>(
        mat.numel(),
        resultData,
        mat.template data<float>(),
        resultData,
        &context_);
  }

  bool RunOnDevice() override {
    auto& predictions = Input(0);
    auto& transitions = Input(1);

    CAFFE_ENFORCE(
        predictions.dim() == 2 && transitions.dim() == 2,
        "Predictions and transitions hould 2D matrices");

    CAFFE_ENFORCE(
        predictions.size(1) == transitions.size(0),
        "Predictions and transitions dimensions not matching");

    auto seqLen = predictions.dim32(0);

    auto* viterbiPath = Output(0, {seqLen}, at::dtype<int32_t>());
    auto block_size = predictions.numel() / predictions.size(0);
    auto block_bytesize =
        predictions.size_from_dim(1) * predictions.dtype().itemsize();
    Tensor backpointers(CPU);
    backpointers.ResizeLike(predictions);

    Tensor trellis(std::vector<int64_t>{block_size}, CPU);
    Tensor dpMat(CPU);
    dpMat.ResizeLike(transitions);
    Tensor dpMax(std::vector<int64_t>{block_size}, CPU);
    GatherRow(predictions, 0, block_size, block_bytesize, &trellis);
    for (auto i = 1; i < seqLen; i++) {
      AddColToMat(transitions, trellis, &dpMat);
      RowwiseMaxAndArg(
          dpMat.template data<float>(),
          dpMat.size(0),
          dpMat.size(1),
          dpMax.template mutable_data<float>(),
          backpointers.template mutable_data<int32_t>() + (i * block_size));

      GatherRow(predictions, i, block_size, block_bytesize, &trellis);
      math::Add<float, CPUContext>(
          trellis.numel(),
          trellis.template data<float>(),
          dpMax.template data<float>(),
          trellis.template mutable_data<float>(),
          &context_);
    }

    Tensor tMax(std::vector<int64_t>{1}, CPU);
    Tensor tArgMax(std::vector<int64_t>{1}, CPU);
    ColwiseMaxAndArg(
        trellis.template data<float>(),
        1,
        trellis.numel(),
        tMax.template mutable_data<float>(),
        tArgMax.template mutable_data<int32_t>());

    std::vector<int32_t> viterbiVec;
    viterbiVec.push_back(tArgMax.template data<int32_t>()[0]);
    Tensor bpEntry(std::vector<int64_t>{block_size}, CPU);
    block_bytesize =
        backpointers.size_from_dim(1) * backpointers.dtype().itemsize();
    for (auto i = seqLen - 1; i > 0; i--) {
      GatherRow(backpointers, i, block_size, block_bytesize, &bpEntry);
      viterbiVec.push_back(bpEntry.template data<int32_t>()[viterbiVec.back()]);
    }
    std::reverse_copy(
        viterbiVec.begin(),
        viterbiVec.end(),
        viterbiPath->template mutable_data<int32_t>());
    return true;
  }
};
class SwapBestPathOp : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit SwapBestPathOp(Args&&... args)
      : Operator(std::forward<Args>(args)...) {}
  bool RunOnDevice() override {
    auto& data = Input(0);
    auto& newBestIdicies = Input(1);

    CAFFE_ENFORCE(
        data.dim() == 2 && newBestIdicies.dim() == 1,
        "predictions should be a 2D matrix and  bestPath should be 1D vector");

    CAFFE_ENFORCE(
        data.size(0) == newBestIdicies.size(0),
        "predictions and bestPath dimensions not matching");

    auto* updatedData = Output(0, data.sizes(), at::dtype<float>());
    float* outData = updatedData->template mutable_data<float>();
    context_.CopyItemsSameDevice(
        data.dtype(), data.numel(), data.template data<float>(), outData);

    Tensor bestScores(CPU);
    bestScores.ResizeLike(newBestIdicies);
    Tensor oldBestIndices(CPU);
    oldBestIndices.ResizeLike(newBestIdicies);

    ColwiseMaxAndArg(
        data.template data<float>(),
        data.size(0),
        data.size(1),
        bestScores.template mutable_data<float>(),
        oldBestIndices.template mutable_data<int32_t>());

    auto block_size = data.numel() / data.size(0);

    const int32_t* oldBestIdx = oldBestIndices.template data<int32_t>();
    const int32_t* newIdx = newBestIdicies.template data<int32_t>();

    for (auto i = 0; i < data.dim32(0); i++) {
      std::swap(
          outData[i * block_size + newIdx[i]],
          outData[i * block_size + oldBestIdx[i]]);
    }
    return true;
  }
};
REGISTER_CPU_OPERATOR(ViterbiPath, ViterbiPathOp);
OPERATOR_SCHEMA(ViterbiPath)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given a predictions matrix and a transitions matrix, get the path with the best
score
)DOC")
    .Input(0, "predictions", "N*D predictions matrix")
    .Input(1, "transitions", "D*D transitions matrix")
    .Output(0, "viterbi_path", "N*1 vector holds the best path indices");
NO_GRADIENT(ViterbiPath);
REGISTER_CPU_OPERATOR(SwapBestPath, SwapBestPathOp);
OPERATOR_SCHEMA(SwapBestPath)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given a sequence of indices and a matrix, enforce that these indices have the
best columnwise scores
score
)DOC")
    .Input(0, "predictions", "N*D predictions matrix")
    .Input(1, "bestPath", "N*1 vector holds the best path indices ")
    .Output(0, "new_predictions", "N*D updated predictions matrix");
NO_GRADIENT(SwapBestPath);
} // namespace
} // namespace caffe2
