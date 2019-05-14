#ifndef CAFFE2_OPERATORS_ASSERT_OP_H_
#define CAFFE2_OPERATORS_ASSERT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class AssertOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit AssertOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        error_msg_(
            this->template GetSingleArgument<std::string>("error_msg", "")) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <typename T>
  bool DoRunWithType() {
    // Copy into CPU context for comparison
    cmp_tensor_.CopyFrom(Input(0));
    auto* cmp_data = cmp_tensor_.template data<T>();

    for (int64_t i = 0; i < cmp_tensor_.numel(); ++i) {
      CAFFE_ENFORCE((bool)cmp_data[i], [&]() {
        std::stringstream ss;
        ss << "Assert failed for element " << i
           << " in tensor, value: " << cmp_data[i] << "\n";
        if (!error_msg_.empty()) {
          ss << "Error message: " << error_msg_;
        }
        return ss.str();
      }());
    }
    return true;
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<long, int, bool>>::call(this, Input(0));
  }

 private:
  Tensor cmp_tensor_{CPU};
  std::string error_msg_;
};

} // namespace caffe2

#endif /* CAFFE2_OPERATORS_ASSERT_OP_H_ */
