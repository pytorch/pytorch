#ifndef CAFFE2_OPERATORS_FUSED_ROWWISE_RAND_CONVERSION_OPS_H_
#define CAFFE2_OPERATORS_FUSED_ROWWISE_RAND_CONVERSION_OPS_H_

#include <chrono>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/reducer_functors.h"
#include "caffe2/perfkernels/math.h"
#include "caffe2/utils/math.h"

#ifdef CAFFE2_USE_MKL
#include <mkl.h>
#define FUSED_ROWWISE_RANDOM_QUANTIZATION_USE_MKL
#endif

namespace caffe2 {

template <class Context>
class FloatToFusedRandRowwiseQuantizedOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit FloatToFusedRandRowwiseQuantizedOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        bitwidth_(OperatorBase::GetSingleArgument<int32_t>("bitwidth", 8)),
        random_(OperatorBase::GetSingleArgument<bool>("random", true)) {
    CAFFE_ENFORCE(
        bitwidth_ == 1 || bitwidth_ == 2 || bitwidth_ == 4 || bitwidth_ == 8,
        "Unsupported bitwidth");
    if (random_) {
#ifdef FUSED_ROWWISE_RANDOM_QUANTIZATION_USE_MKL
      int status = vslNewStream(
          &vslStream_,
          VSL_BRNG_MT19937,
          std::chrono::system_clock::now().time_since_epoch().count());
      if (status != VSL_STATUS_OK) {
        LOG(WARNING) << "vslNewStream returns " << status;
      }
#else
      gen_.seed(std::chrono::system_clock::now().time_since_epoch().count());
      dis_.reset(new std::uniform_real_distribution<float>(0.0f, 1.0f));
#endif
    }
  }

  ~FloatToFusedRandRowwiseQuantizedOp() {
    if (random_) {
#ifdef FUSED_ROWWISE_RANDOM_QUANTIZATION_USE_MKL
      int status = vslDeleteStream(&vslStream_);
      if (status != VSL_STATUS_OK) {
        LOG(WARNING) << "vslDeleteStream returns " << status;
      }
#endif
    }
  }

  bool RunOnDevice() override;

 private:
  INPUT_TAGS(DATA_FLOAT);
  OUTPUT_TAGS(DATA_FUSED_QUANTIZED);

 protected:
  size_t bitwidth_{8};
  bool random_{true};
  std::vector<float> random_buffer_;

#ifdef FUSED_ROWWISE_RANDOM_QUANTIZATION_USE_MKL
  VSLStreamStatePtr vslStream_;
#else
  std::unique_ptr<std::uniform_real_distribution<float>> dis_;
  std::minstd_rand gen_;
#endif
};

template <class Context>
class FusedRandRowwiseQuantizedToFloatOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(FusedRandRowwiseQuantizedToFloatOp)

  bool RunOnDevice() override;

 private:
  INPUT_TAGS(DATA_FUSED_QUANTIZED);
  OUTPUT_TAGS(DATA_FLOAT);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FUSED_ROWWISE_RAND_CONVERSION_OPS_H_
