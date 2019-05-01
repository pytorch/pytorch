#ifndef CAFFE2_MAP_OP_H_
#define CAFFE2_MAP_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class APMeterOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit APMeterOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        buffer_size_(
            this->template GetSingleArgument<int32_t>("buffer_size", 1000)),
        buffer_used_(0) {}

  bool RunOnDevice() override;

 protected:
  using BufferDataType = std::pair<float, int>;
  // Buffer the predictions for each class
  std::vector<std::vector<BufferDataType>> buffers_;
  // Capacity of the buffer
  int buffer_size_;
  // Used buffer
  int buffer_used_;

  INPUT_TAGS(PREDICTION, LABEL);

 protected:
  // Buffer predictions for N sample and D classes
  void
  BufferPredictions(const float* Xdata, const int* labelData, int N, int D);
};

} // namespace caffe2

#endif // CAFFE2_MAP_OP_H_
