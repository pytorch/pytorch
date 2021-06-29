#include "caffe2/operators/cast_op.h"

namespace caffe2 {

template <typename DstType, typename SrcType>
struct CastHelper {
  static DstType call(SrcType data) {
    return static_cast<DstType>(data);
  }
};

template <typename SrcType>
struct CastHelper<std::string, SrcType> {
  static std::string call(SrcType data) {
    return caffe2::to_string(data);
  }
};

template <>
template <typename DstType, typename SrcType>
bool CastOp<CPUContext>::DoRunWithType() {
  auto& input = Input(0);

  auto* output = Output(0, input.sizes(), at::dtype<DstType>());
  const auto* data = input.template data<SrcType>();
  auto* out = output->template mutable_data<DstType>();
  auto N = input.numel();
  for (int64_t i = 0; i < N; ++i) {
    out[i] = CastHelper<DstType, SrcType>::call(data[i]);
  }
  return true;
}

template <>
void CastOp<CPUContext>::SetBody(TensorProto_DataType to) {
  switch (to) {
    case TensorProto_DataType_FLOAT:
      // body_ = &CastOp::DoRunIncFp16WithDstType<float>;
      body_ = &CastOp<CPUContext>::DoRunWithDstType<float>;
      break;
    case TensorProto_DataType_INT32:
      body_ = &CastOp<CPUContext>::DoRunWithDstType<int>;
      break;
    case TensorProto_DataType_BYTE:
      LOG(FATAL) << "BYTE is deprecated";
      break;
    case TensorProto_DataType_STRING:
      body_ = &CastOp<CPUContext>::DoRunWithDstType<std::string>;
      break;
    case TensorProto_DataType_BOOL:
      body_ = &CastOp<CPUContext>::DoRunWithDstType<bool>;
      break;
    case TensorProto_DataType_UINT8:
      body_ = &CastOp<CPUContext>::DoRunWithDstType<uint8_t>;
      break;
    case TensorProto_DataType_INT8:
      body_ = &CastOp<CPUContext>::DoRunWithDstType<int8_t>;
      break;
    case TensorProto_DataType_UINT16:
      body_ = &CastOp<CPUContext>::DoRunWithDstType<uint16_t>;
      break;
    case TensorProto_DataType_INT16:
      body_ = &CastOp<CPUContext>::DoRunWithDstType<int16_t>;
      break;
    case TensorProto_DataType_INT64:
      body_ = &CastOp<CPUContext>::DoRunWithDstType<int64_t>;
      break;
    case TensorProto_DataType_FLOAT16:
      CAFFE_THROW("Casting to and from at::Half on CPU is not supported yet");
      // break;
    case TensorProto_DataType_DOUBLE:
      // body_ = &CastOp::DoRunIncFp16WithDstType<double>;
      body_ = &CastOp<CPUContext>::DoRunWithDstType<double>;
      break;
    case TensorProto_DataType_UNDEFINED:
      CAFFE_THROW("Cast op must have 'to' argument of type DataType");
      // break;
    default:
      CAFFE_THROW("Unexpected 'to' argument value: ", to);
  }
}

template <>
template <typename DstType>
bool CastOp<CPUContext>::DoRunWithDstType() {
  return DispatchHelper<
      TensorTypes<
          float,
          int32_t,
          bool,
          uint8_t,
          int8_t,
          uint16_t,
          int16_t,
          int64_t,
          double>,
      DstType>::call(this, Input(0));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Cast, CastOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Cast)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      vector<TensorShape> out;
      out.push_back(in[0]);
      out[0].set_data_type(cast::GetCastDataType(helper, "to"));
      return out;
    })
    .SetDoc(R"DOC(
Casts the elements of a given input tensor to a data type specified by the `to`
argument and returns an output tensor of the same size in the converted type.
The `to` argument must be one of the data types specified in the *DataType*
enum field in the TensorProto message (see below). If the `to` argument is not
provided or is not one of the enumerated types in *DataType*, Caffe2 throws an
Enforce error.

NOTE: Casting from strings is not supported, and casting to strings is only
supported on CPU.

TensorProto *DataType* field:
```
message TensorProto {
  ...
  enum DataType {
    UNDEFINED = 0;
    FLOAT = 1;  // float
    INT32 = 2;  // int
    BYTE = 3;  // BYTE, when deserialized, is going to be restored as uint8.
    STRING = 4;  // string
    BOOL = 5;  // bool
    UINT8 = 6;  // uint8_t
    INT8 = 7;  // int8_t
    UINT16 = 8;  // uint16_t
    INT16 = 9;  // int16_t
    INT64 = 10;  // int64_t
    FLOAT16 = 12;  // at::Half
    DOUBLE = 13;  // double
  }
```

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cast_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "Cast",
    ["X"],
    ["Y"],
    to=2
)

workspace.FeedBlob("X", (np.random.rand(3,3)).astype(np.float32)*10)
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))
```

**Result**

```
X: [[9.436466   5.8529844  0.54932857]
 [1.1583444  2.9936118  0.22950427]
 [3.9143739  3.4040766  8.905341  ]]
Y: [[9 5 0]
 [1 2 0]
 [3 3 8]]
```

</details>

)DOC")
    .Arg(
        "to",
        "*(type: int)* Data type to which the elements of the input tensor are "
        "cast. Strictly must be one of the types from *DataType* enum in "
        "TensorProto.")
    .Input(0, "X", "*(type: Tensor)* Input tensor to be cast.")
    .Output(
        0,
        "Y",
        "*(type: Tensor`<'to' type>`)* Output tensor with the same shape as "
        "input with type specified by the `to` argument.")
    .InheritOnnxSchema();

// Some Casts are compatible with gradients, but for now we don't support it
// GRADIENT_NOT_IMPLEMENTED_YET(Cast);

class GetCastGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {

    vector<OperatorDef> defs = SingleGradientDef("Cast", "", vector<string>{GO(0)}, vector<string>{GI(0)});

    // now modify the arguments in defs[0]
    ArgumentHelper argsHelper(def_);

    auto to_name = cast::GetCastDataType(argsHelper, "to");

    CAFFE_ENFORCE(
        argsHelper.HasSingleArgumentOfType<string>("from_type") ||
            argsHelper.HasSingleArgumentOfType<int>("from_type"),
        "Argument 'from_type' of type int or string"
        " is required to get the gradient of CastOp");

    auto from_name = cast::GetCastDataType(argsHelper, "from_type");
    Argument *to = defs[0].add_arg();
    to->set_name("to");
    to->set_i(from_name);

    Argument *from = defs[0].add_arg();
    from->set_name("from_type");
    from->set_i(to_name);

    return defs;
  }

  bool CopyArguments() const override {
    return false;
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Cast, GetCastGradient);




}  // namespace caffe2
