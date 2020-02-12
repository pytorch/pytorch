#ifndef CAFFE2_OPERATORS_INDEX_HASH_OPS_H_
#define CAFFE2_OPERATORS_INDEX_HASH_OPS_H_

#include "caffe2/core/asan.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(IndexHash);

namespace caffe2 {

template <class Context>
class IndexHashOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit IndexHashOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        seed_(this->template GetSingleArgument<int64_t>("seed", 0)),
        modulo_(this->template GetSingleArgument<int64_t>("modulo", 0)) {
    CAFFE_ENFORCE_GT(modulo_, 0, "MODULO should be > 0");
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& indices = Input(INDICES);

    auto* hashed_indices =
        Output(HASHED_INDICES, indices.sizes(), at::dtype<T>());

    CAFFE_ENFORCE_GE(
        static_cast<int64_t>(std::numeric_limits<T>::max()),
        modulo_,
        "MODULO shouldn't be larger than the numeric limit of the indices");

    auto N = indices.numel();
    auto* indices_data = indices.template data<T>();
    auto* hashed_indices_data = hashed_indices->template mutable_data<T>();

    for (auto i = 0; i < N; i++) {
      hashed_indices_data[i] = hash(indices_data[i]);
    }

    return true;
  }

 protected:
  template <typename T>
  CAFFE2_NO_SANITIZE("signed-integer-overflow")
  T hash(T id) {
    int8_t* bytes = (int8_t*)&id;
    T hashed = seed_ * 0xDEADBEEF;
    for (int i = 0; i < sizeof(T) / sizeof(int8_t); i++) {
      hashed = hashed * 65537 + bytes[i];
    }
    // We want the result of the modulo to be positive. This works under the
    // assumption that modulo_ > 0 which is enforced in the constructor.
    auto modHashed = hashed % modulo_;
    return modHashed >= 0 ? modHashed : modHashed + modulo_;
  }

 private:
  INPUT_TAGS(INDICES);
  OUTPUT_TAGS(HASHED_INDICES);

  int64_t seed_;
  int64_t modulo_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INDEX_HASH_OPS_H_
