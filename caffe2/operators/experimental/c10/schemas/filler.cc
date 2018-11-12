#include "caffe2/operators/experimental/c10/schemas/filler.h"
#include "caffe2/core/dispatch/OpSchemaRegistration.h"
#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/utils/cast.h"

using caffe2::CPUContext;
using caffe2::Tensor;

C10_DEFINE_OP_SCHEMA(caffe2::ops::ConstantFill);
C10_DEFINE_OP_SCHEMA(caffe2::ops::UniformFill);

C10_DEFINE_OP_SCHEMA(caffe2::ops::GivenTensorFill<float>);
C10_DEFINE_OP_SCHEMA(caffe2::ops::GivenTensorFill<int>);
C10_DEFINE_OP_SCHEMA(caffe2::ops::GivenTensorFill<int64_t>);

namespace {
struct ShapeParameter final {
  using type = std::vector<int64_t>;
  static std::vector<int64_t> parse(const caffe2::ArgumentHelper& helper) {
    return helper.GetRepeatedArgument<int64_t>("shape");
  }
};
struct ExtraShapeParameter final {
  using type = std::vector<int>;
  static std::vector<int> parse(const caffe2::ArgumentHelper& helper) {
    return helper.GetRepeatedArgument<int>("extra_shape");
  }
};
struct InputAsShapeParameter final {
  using type = bool;
  static bool parse(const caffe2::ArgumentHelper& helper) {
    return helper.GetSingleArgument<bool>("input_as_shape", false);
  }
};
struct DTypeParameter final {
  using type = int;
  static int parse(const caffe2::ArgumentHelper& helper) {
    auto dtype = helper.GetSingleArgument<int>(
        "dtype", caffe2::TensorProto_DataType_FLOAT);
    if (!helper.HasArgument("dtype") && helper.HasArgument("value")) {
      // If 'dtype' is not provided, infer type based on the type of 'value'
      // Currently, single argument contains either float, int64 or bytes
      if (helper.HasSingleArgumentOfType<float>("value")) {
        dtype = caffe2::TensorProto_DataType_FLOAT;
      } else if (helper.HasSingleArgumentOfType<int64_t>("value")) {
        dtype = caffe2::TensorProto_DataType_INT64;
      } else {
        CAFFE_THROW("Argument 'value' is of unexpected type");
      }
      VLOG(1) << "Argument 'dtype' is not provided. Assume the data type is "
              << "the same as that of argument 'value': " << dtype;
    }
    return dtype;
  }
};
struct ValueParameter final {
  using type = caffe2::ops::ConstantFill::Value;
  static caffe2::ops::ConstantFill::Value parse(
      const caffe2::ArgumentHelper& helper) {
    caffe2::ops::ConstantFill::Value result;
    if (helper.HasSingleArgumentOfType<float>("value")) {
      result.as_float = helper.GetSingleArgument<float>("value", 0);
    } else if (helper.HasSingleArgumentOfType<int32_t>("value")) {
      result.as_int32 = helper.GetSingleArgument<int32_t>("value", 0);
    } else if (helper.HasSingleArgumentOfType<int64_t>("value")) {
      result.as_int64 = helper.GetSingleArgument<int64_t>("value", 0);
    } else if (helper.HasSingleArgumentOfType<bool>("value")) {
      result.as_bool = helper.GetSingleArgument<bool>("value", false);
    } else {
      result.as_float = 0.0;
    }
    return result;
  }
};
struct MinParameter final {
  using type = float;
  static float parse(const caffe2::ArgumentHelper& helper) {
    return helper.GetSingleArgument<float>("min", 0);
  }
};
struct MaxParameter final {
  using type = float;
  static float parse(const caffe2::ArgumentHelper& helper) {
    return helper.GetSingleArgument<float>("max", 1);
  }
};
template <class T>
struct ValuesParameter final {
  using type = Tensor;
  static Tensor parse(const caffe2::ArgumentHelper& helper) {
    if (!std::is_same<T, float>::value || !helper.HasArgument("dtype")) {
      return ExtractValues<T>(helper);
    } else {
      auto dtype = caffe2::cast::GetCastDataType(helper, "dtype");
      switch (dtype) {
        case caffe2::TensorProto_DataType_FLOAT:
          return ExtractValues<float>(helper);
        case caffe2::TensorProto_DataType_DOUBLE:
          return ExtractValues<double>(helper);
        case caffe2::TensorProto_DataType_BOOL:
          return ExtractValues<bool>(helper);
        case caffe2::TensorProto_DataType_INT32:
          return ExtractValues<int>(helper);
        case caffe2::TensorProto_DataType_INT64:
          return ExtractValues<int64_t>(helper);
        case caffe2::TensorProto_DataType_STRING:
          return ExtractValues<std::string>(helper);
        case caffe2::TensorProto_DataType_UNDEFINED:
          CAFFE_THROW("Cannot have undefined 'dtype' argument");
        default:
          CAFFE_THROW("Unexpected 'dtype' argument value: ", dtype);
      }
    }
  }

 private:
  template <typename Type>
  static Tensor ExtractValues(
      const caffe2::ArgumentHelper& helper) {
    auto source_values = helper.GetRepeatedArgument<Type>("values");
    Tensor values{caffe2::CPU};
    values.Resize(source_values.size());
    Type* values_data = values.template mutable_data<Type>();
    for (int i = 0; i < source_values.size(); i++) {
      values_data[i] = static_cast<Type>(source_values[i]);
    }
    // body_ = &GivenTensorFillOp::FillWithType<Type>;
    return values;
  }
};
} // namespace

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_ARRAY_INPUT_AND_PARAMETERS(
    ops::ConstantFill,
    void,
    C10ConstantFill_DontUseThisOpYet,
    ShapeParameter,
    ExtraShapeParameter,
    InputAsShapeParameter,
    DTypeParameter,
    ValueParameter)
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_ARRAY_INPUT_AND_PARAMETERS(
    ops::UniformFill,
    void,
    C10UniformFill_DontUseThisOpYet,
    ShapeParameter,
    ExtraShapeParameter,
    InputAsShapeParameter,
    MinParameter,
    MaxParameter)

REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_ARRAY_INPUT_AND_PARAMETERS(
    ops::GivenTensorFill<float>,
    void,
    C10GivenTensorFill_DontUseThisOpYet,
    ShapeParameter,
    ExtraShapeParameter,
    InputAsShapeParameter,
    ValuesParameter<float>)
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_ARRAY_INPUT_AND_PARAMETERS(
    ops::GivenTensorFill<int>,
    void,
    C10GivenTensorIntFill_DontUseThisOpYet,
    ShapeParameter,
    ExtraShapeParameter,
    InputAsShapeParameter,
    ValuesParameter<int>)
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_ARRAY_INPUT_AND_PARAMETERS(
    ops::GivenTensorFill<int64_t>,
    void,
    C10GivenTensorInt64Fill_DontUseThisOpYet,
    ShapeParameter,
    ExtraShapeParameter,
    InputAsShapeParameter,
    ValuesParameter<int64_t>)
} // namespace caffe2
