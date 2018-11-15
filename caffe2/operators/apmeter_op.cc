#include "caffe2/operators/apmeter_op.h"

namespace caffe2 {

template <>
void APMeterOp<float, CPUContext>::BufferPredictions(
    const float* XData,
    const int* labelData,
    int N,
    int D) {
  if (buffers_.empty()) {
    // Initialize the buffer
    buffers_.resize(D, std::vector<BufferDataType>(buffer_size_));
  }
  DCHECK_EQ(buffers_.size(), D);

  // Fill atmose buffer_size_ data at a time, so truncate the input if needed
  if (N > buffer_size_) {
    XData = XData + (N - buffer_size_) * D;
    labelData = labelData + (N - buffer_size_) * D;
    N = buffer_size_;
  }

  // Reclaim space if not enough space in the buffer to hold new data
  int space_to_reclaim = buffer_used_ + N - buffer_size_;
  if (space_to_reclaim > 0) {
    for (auto& buffer : buffers_) {
      std::rotate(
          buffer.begin(), buffer.begin() + space_to_reclaim, buffer.end());
    }
    buffer_used_ -= space_to_reclaim;
  }

  // Fill the buffer
  for (int i = 0; i < D; i++) {
    for (int j = 0; j < N; j++) {
      buffers_[i][buffer_used_ + j].first = XData[j * D + i];
      buffers_[i][buffer_used_ + j].second = labelData[j * D + i];
    }
  }

  buffer_used_ += N;
}

template <>
bool APMeterOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(PREDICTION);
  auto& label = Input(LABEL);
  auto* Y = Output(0);
  // Check dimensions
  DCHECK_EQ(X.dim(), 2);
  int N = X.dim32(0);
  int D = X.dim32(1);
  DCHECK_EQ(label.dim(), 2);
  DCHECK_EQ(label.dim32(0), N);
  DCHECK_EQ(label.dim32(1), D);
  Y->Resize(D);

  const auto* Xdata = X.data<float>();
  const auto* labelData = label.data<int>();
  auto* Ydata = Y->template mutable_data<float>();

  BufferPredictions(Xdata, labelData, N, D);

  // Calculate AP for each class
  for (int i = 0; i < D; i++) {
    auto& buffer = buffers_[i];
    // Sort predictions by score
    std::stable_sort(
        buffer.begin(),
        buffer.begin() + buffer_used_,
        [](const BufferDataType& p1, const BufferDataType& p2) {
          return p1.first > p2.first;
        });
    // Calculate cumulative precision for each sample
    float tp_sum = 0.0;
    float precision_sum = 0.0;
    int ntruth = 0;
    for (int j = 0; j < buffer_used_; j++) {
      tp_sum += buffer[j].second;
      if (buffer[j].second == 1) {
        ntruth += 1;
        precision_sum += tp_sum / (j + 1);
      }
    }

    // Calculate AP
    Ydata[i] = precision_sum / std::max(1, ntruth);
  }

  return true;
}

namespace {
REGISTER_CPU_OPERATOR(APMeter, APMeterOp<float, CPUContext>);

OPERATOR_SCHEMA(APMeter)
    .NumInputs(2)
    .NumOutputs(1)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc(R"DOC(
APMeter computes Average Precision for binary or multi-class classification.
It takes two inputs: prediction scores P of size (n_samples x n_classes), and
true labels Y of size (n_samples x n_classes). It returns a single float number
per class for the average precision of that class.
)DOC")
    .Arg(
        "buffer_size",
        "(int32_t) indicates how many predictions should the op buffer. "
        "defaults to 1000")
    .Input(
        0,
        "predictions",
        "2-D tensor (Tensor<float>) of size (num_samples x"
        "num_classes) containing prediction scores")
    .Input(
        1,
        "labels",
        "2-D tensor (Tensor<float>) of size (num_samples) "
        "containing true labels for each sample")
    .Output(
        0,
        "AP",
        "1-D tensor (Tensor<float>) of size num_classes containing "
        "average precision for each class");

SHOULD_NOT_DO_GRADIENT(APMeter);

} // namespace
} // namespace caffe2
