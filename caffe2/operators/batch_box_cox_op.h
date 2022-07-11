#ifndef CAFFE_OPERATORS_BATCH_BOX_COX_OPS_H_
#define CAFFE_OPERATORS_BATCH_BOX_COX_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(BatchBoxCox);

namespace caffe2 {

template <class Context>
class BatchBoxCoxOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit BatchBoxCoxOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        min_block_size_(
            this->template GetSingleArgument<int>("min_block_size", 256)) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(DATA));
  }

  template <typename T>
  bool DoRunWithType();

 protected:
  template <typename T>
  void BoxCoxNaive(
      int64_t N,
      int64_t D,
      const T* data_ptr,
      const T* lambda1_ptr,
      const T* lambda2_ptr,
      T k_eps,
      T* output_ptr);

#ifdef CAFFE2_USE_MKL
  template <typename T>
  void BoxCoxNonzeroLambda(
      int64_t D,
      const T* data_ptr,
      const T* lambda1,
      const T* lambda2,
      T k_eps,
      T* output_ptr);

  template <typename T>
  void BoxCoxZeroLambda(
      int64_t D,
      const T* data_ptr,
      const T* lambda2,
      T k_eps,
      T* output_ptr);

  template <typename T>
  void BoxCoxMixedLambda(
      const T* data_ptr,
      const vector<int>& nonzeros,
      const vector<int>& zeros,
      const T* lambda1,
      const T* lambda2,
      const T* lambda2_z,
      T k_eps,
      T* buffer,
      T* output_ptr);

  vector<int> nonzeros_, zeros_;

  // Buffers used by the MKL version are cached across calls.
  struct CachedBuffers {
    virtual ~CachedBuffers() {}
    int type_;
  };
  template <typename T>
  struct TypedCachedBuffers : public CachedBuffers {
    vector<T> lambda1_, lambda2_, lambda2_z_;
    vector<T> accumulator_;
  };
  template <typename T>
  TypedCachedBuffers<T>& GetBuffers();
  unique_ptr<CachedBuffers> buffers_;

#endif // CAFFE2_USE_MKL

  int min_block_size_;

  INPUT_TAGS(DATA, LAMBDA1, LAMBDA2);
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_BATCH_BOX_COX_OPS_H_
