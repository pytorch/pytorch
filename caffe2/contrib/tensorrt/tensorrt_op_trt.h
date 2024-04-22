#pragma once

#include "caffe2/contrib/tensorrt/trt_utils.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"

#include <NvInfer.h>
#include <unordered_map>

namespace caffe2 {

class TensorRTOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  TensorRTOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;
  virtual ~TensorRTOp() noexcept {}

 private:
  void MaybeAdjustOutputShape(int output_idx, std::vector<int64_t>* dims);

  tensorrt::TrtLogger logger_;
  int max_batch_size_;
  std::vector<nvinfer1::Dims> nv_dims_;
  std::vector<bool> is_input_;
  std::unordered_map<int, std::vector<int64_t>> output_size_hints_;
  std::shared_ptr<nvinfer1::ICudaEngine> trt_engine_{nullptr};
  std::shared_ptr<nvinfer1::IExecutionContext> trt_executor_{nullptr};
  bool batch_warning_issued_{false};
};

} // namespace caffe2

