#ifndef CAFFE2_OPERATORS_FIND_OP_H_
#define CAFFE2_OPERATORS_FIND_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "c10/util/irange.h"

#include <unordered_map>

namespace caffe2 {

template <class Context>
class FindOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit FindOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        missing_value_(
            this->template GetSingleArgument<int>("missing_value", -1)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_DISPATCH_HELPER;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int, long>>::call(this, Input(0));
  }

 protected:
  template <typename T>
  bool DoRunWithType() {
    auto& idx = Input(0);
    auto& needles = Input(1);

    auto* res_indices = Output(0, needles.sizes(), at::dtype<T>());

    const T* idx_data = idx.template data<T>();
    const T* needles_data = needles.template data<T>();
    T* res_data = res_indices->template mutable_data<T>();
    auto idx_size = idx.numel();

    // Use an arbitrary cut-off for when to use brute-force
    // search. For larger needle sizes we first put the
    // index into a map
    if (needles.numel() < 16) {
      // Brute force O(nm)
      for (const auto i : c10::irange(needles.numel())) {
        T x = needles_data[i];
        T res = static_cast<T>(missing_value_);
        for (int j = idx_size - 1; j >= 0; j--) {
          if (idx_data[j] == x) {
            res = j;
            break;
          }
        }
        res_data[i] = res;
      }
    } else {
      // O(n + m)
      std::unordered_map<T, int> idx_map;
      for (const auto j : c10::irange(idx_size)) {
        idx_map[idx_data[j]] = j;
      }
      for (const auto i : c10::irange(needles.numel())) {
        T x = needles_data[i];
        auto it = idx_map.find(x);
        res_data[i] = (it == idx_map.end() ? missing_value_ : it->second);
      }
    }

    return true;
  }

 protected:
  int missing_value_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FIND_OP_H_
