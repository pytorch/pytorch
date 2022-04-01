#ifndef CAFFE2_OPERATORS_ELEMENTWISE_LOGICAL_OPS_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_LOGICAL_OPS_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/elementwise_ops.h"

#include <unordered_set>

namespace caffe2 {

template <class Context>
class WhereOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);
  USE_DISPATCH_HELPER;

  template <class... Args>
  explicit WhereOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(bool, "broadcast_on_rows", enable_broadcast_, 0) {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<float, double, int, long, std::string, bool>>::
        call(this, Input(1));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& select = Input(0);
    auto& left = Input(1);
    auto& right = Input(2);

    if (enable_broadcast_) {
      CAFFE_ENFORCE_EQ(select.dim(), 1);
      CAFFE_ENFORCE_EQ(select.size(0), right.size(0));
      CAFFE_ENFORCE_EQ(left.sizes(), right.sizes());
    } else {
      CAFFE_ENFORCE_EQ(select.sizes(), left.sizes());
      CAFFE_ENFORCE_EQ(select.sizes(), right.sizes());
    }
    auto* output = Output(0, left.sizes(), at::dtype<T>());

    const bool* select_data = select.template data<bool>();
    const T* left_data = left.template data<T>();
    const T* right_data = right.template data<T>();
    T* output_data = output->template mutable_data<T>();

    if (enable_broadcast_) {
      size_t block_size = left.size_from_dim(1);
      for (const auto i : c10::irange(select.numel())) {
        size_t offset = i * block_size;
        if (select_data[i]) {
          context_.CopyItemsSameDevice(
              output->dtype(),
              block_size,
              left_data + offset,
              output_data + offset);
        } else {
          context_.CopyItemsSameDevice(
              output->dtype(),
              block_size,
              right_data + offset,
              output_data + offset);
        }
      }
    } else {
      for (const auto i : c10::irange(select.numel())) {
        output_data[i] = select_data[i] ? left_data[i] : right_data[i];
      }
    }
    return true;
  }

 private:
  bool enable_broadcast_;
};

class IsMemberOfValueHolder {
  std::unordered_set<int32_t> int32_values_;
  std::unordered_set<int64_t> int64_values_;
  std::unordered_set<bool> bool_values_;
  std::unordered_set<std::string> string_values_;
  bool has_values_ = false;

 public:
  template <typename T>
  std::unordered_set<T>& get();

  template <typename T>
  void set(const std::vector<T>& args) {
    has_values_ = true;
    auto& values = get<T>();
    values.insert(args.begin(), args.end());
  }

  bool has_values() {
    return has_values_;
  }
};

template <class Context>
class IsMemberOfOp final : public Operator<Context> {
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_DISPATCH_HELPER;

  static constexpr const char* VALUE_TAG = "value";

 public:
  using TestableTypes = TensorTypes<int32_t, int64_t, bool, std::string>;

  template <class... Args>
  explicit IsMemberOfOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    auto dtype =
        static_cast<TensorProto_DataType>(this->template GetSingleArgument<int>(
            "dtype", TensorProto_DataType_UNDEFINED));
    switch (dtype) {
      case TensorProto_DataType_INT32:
        values_.set(this->template GetRepeatedArgument<int32_t>(VALUE_TAG));
        break;
      case TensorProto_DataType_INT64:
        values_.set(this->template GetRepeatedArgument<int64_t>(VALUE_TAG));
        break;
      case TensorProto_DataType_BOOL:
        values_.set(this->template GetRepeatedArgument<bool>(VALUE_TAG));
        break;
      case TensorProto_DataType_STRING:
        values_.set(this->template GetRepeatedArgument<std::string>(VALUE_TAG));
        break;
      case TensorProto_DataType_UNDEFINED:
        // If dtype is not provided, values_ will be filled the first time that
        // DoRunWithType is called.
        break;
      default:
        CAFFE_THROW("Unexpected 'dtype' argument value: ", dtype);
    }
  }
  virtual ~IsMemberOfOp() noexcept {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<int32_t, int64_t, bool, std::string>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& input = Input(0);

    auto* output = Output(0, input.sizes(), at::dtype<bool>());

    if (!values_.has_values()) {
      values_.set(this->template GetRepeatedArgument<T>(VALUE_TAG));
    }
    const auto& values = values_.get<T>();

    const T* input_data = input.template data<T>();
    bool* output_data = output->template mutable_data<bool>();
    for (const auto i : c10::irange(input.numel())) {
      output_data[i] = values.find(input_data[i]) != values.end();
    }
    return true;
  }

 protected:
  IsMemberOfValueHolder values_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ELEMENTWISE_LOGICAL_OPS_H_
