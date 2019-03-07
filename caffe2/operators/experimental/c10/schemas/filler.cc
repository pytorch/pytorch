#include "caffe2/operators/experimental/c10/schemas/filler.h"
#include <ATen/core/dispatch/OpSchemaRegistration.h>
#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/utils/cast.h"

using caffe2::CPUContext;
using c10::C10Tensor;
using c10::ivalue::IntList;
using c10::intrusive_ptr;

namespace caffe2 {
namespace ops {
// TODO Parse schema strings instead of creating FunctionSchema manually
C10_DEFINE_OP_SCHEMA(ConstantFill, FunctionSchema(
    "_c10_experimental::ConstantFill",
    "",
    (std::vector<c10::Argument>{
      c10::Argument("inputs", ListType::ofTensors()),
      c10::Argument("output"),
      c10::Argument("shape", ListType::ofInts()),
      c10::Argument("extra_shape", ListType::ofInts()),
      c10::Argument("input_as_shape", BoolType::get()),
      c10::Argument("dtype", IntType::get()),
      c10::Argument("value", NumberType::get())
    }), (std::vector<c10::Argument>{
    })
));
C10_DEFINE_OP_SCHEMA(UniformFill, FunctionSchema(
    "_c10_experimental::UniformFill",
    "",
    (std::vector<c10::Argument>{
      c10::Argument("inputs", ListType::ofTensors()),
      c10::Argument("output"),
      c10::Argument("shape", ListType::ofInts()),
      c10::Argument("extra_shape", ListType::ofInts()),
      c10::Argument("input_as_shape", BoolType::get()),
      c10::Argument("min", FloatType::get()),
      c10::Argument("max", FloatType::get())
    }), (std::vector<c10::Argument>{
    })
));
C10_DEFINE_OP_SCHEMA(GivenTensorFill, FunctionSchema(
    "_c10_experimental::GivenTensorFill",
    "",
    (std::vector<c10::Argument>{
      c10::Argument("inputs", ListType::ofTensors()),
      c10::Argument("output"),
      c10::Argument("shape", ListType::ofInts()),
      c10::Argument("extra_shape", ListType::ofInts()),
      c10::Argument("input_as_shape", BoolType::get()),
      c10::Argument("values"),
    }), (std::vector<c10::Argument>{
    })
));
C10_DEFINE_OP_SCHEMA(GivenTensorIntFill, FunctionSchema(
    "_c10_experimental::GivenTensorIntFill",
    "",
    (std::vector<c10::Argument>{
      c10::Argument("inputs", ListType::ofTensors()),
      c10::Argument("output"),
      c10::Argument("shape", ListType::ofInts()),
      c10::Argument("extra_shape", ListType::ofInts()),
      c10::Argument("input_as_shape", BoolType::get()),
      c10::Argument("values"),
    }), (std::vector<c10::Argument>{
    })
));
C10_DEFINE_OP_SCHEMA(GivenTensorInt64Fill, FunctionSchema(
    "_c10_experimental::GivenTensorInt64Fill",
    "",
    (std::vector<c10::Argument>{
      c10::Argument("inputs", ListType::ofTensors()),
      c10::Argument("output"),
      c10::Argument("shape", ListType::ofInts()),
      c10::Argument("extra_shape", ListType::ofInts()),
      c10::Argument("input_as_shape", BoolType::get()),
      c10::Argument("values"),
    }), (std::vector<c10::Argument>{
    })
));
}
}

namespace {
struct ShapeParameter final {
  using type = intrusive_ptr<IntList>;
  static intrusive_ptr<IntList> parse(const caffe2::ArgumentHelper& helper) {
    return IntList::create(helper.GetRepeatedArgument<int64_t>("shape"));
  }
};
struct ExtraShapeParameter final {
  using type = intrusive_ptr<IntList>;
  static intrusive_ptr<IntList> parse(const caffe2::ArgumentHelper& helper) {
    return IntList::create(helper.GetRepeatedArgument<int64_t>("extra_shape"));
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
  using type = c10::IValue;
  static c10::IValue parse(
      const caffe2::ArgumentHelper& helper) {
    c10::IValue result;
    if (helper.HasSingleArgumentOfType<float>("value")) {
      result = helper.GetSingleArgument<float>("value", 0);
    } else if (helper.HasSingleArgumentOfType<int32_t>("value")) {
      result = helper.GetSingleArgument<int32_t>("value", 0);
    } else if (helper.HasSingleArgumentOfType<int64_t>("value")) {
      result = helper.GetSingleArgument<int64_t>("value", 0);
    } else if (helper.HasSingleArgumentOfType<bool>("value")) {
      result = helper.GetSingleArgument<bool>("value", false);
    } else {
      result = 0.0;
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
  using type = at::Tensor;
  static at::Tensor parse(const caffe2::ArgumentHelper& helper) {
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
  static at::Tensor ExtractValues(
      const caffe2::ArgumentHelper& helper) {
    auto source_values = helper.GetRepeatedArgument<Type>("values");
    caffe2::Tensor values{caffe2::CPU};
    values.Resize(source_values.size());
    Type* values_data = values.template mutable_data<Type>();
    for (int i = 0; i < source_values.size(); i++) {
      values_data[i] = static_cast<Type>(source_values[i]);
    }
    // body_ = &GivenTensorFillOp::FillWithType<Type>;
    return at::Tensor(C10Tensor(values));
  }
};
} // namespace

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_ARRAY_INPUT_AND_PARAMETERS(
    ops::ConstantFill,
    C10ConstantFill_DontUseThisOpYet,
    1,
    ShapeParameter,
    ExtraShapeParameter,
    InputAsShapeParameter,
    DTypeParameter,
    ValueParameter)
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_ARRAY_INPUT_AND_PARAMETERS(
    ops::UniformFill,
    C10UniformFill_DontUseThisOpYet,
    1,
    ShapeParameter,
    ExtraShapeParameter,
    InputAsShapeParameter,
    MinParameter,
    MaxParameter)

REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_ARRAY_INPUT_AND_PARAMETERS(
    ops::GivenTensorFill,
    C10GivenTensorFill_DontUseThisOpYet,
    1,
    ShapeParameter,
    ExtraShapeParameter,
    InputAsShapeParameter,
    ValuesParameter<float>)
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_ARRAY_INPUT_AND_PARAMETERS(
    ops::GivenTensorIntFill,
    C10GivenTensorIntFill_DontUseThisOpYet,
    1,
    ShapeParameter,
    ExtraShapeParameter,
    InputAsShapeParameter,
    ValuesParameter<int>)
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_ARRAY_INPUT_AND_PARAMETERS(
    ops::GivenTensorInt64Fill,
    C10GivenTensorInt64Fill_DontUseThisOpYet,
    1,
    ShapeParameter,
    ExtraShapeParameter,
    InputAsShapeParameter,
    ValuesParameter<int64_t>)
} // namespace caffe2
