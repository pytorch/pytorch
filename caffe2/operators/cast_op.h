#ifndef CAFFE2_OPERATORS_CAST_OP_H_
#define CAFFE2_OPERATORS_CAST_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace cast {

inline TensorProto_DataType GetCastDataType(const ArgumentHelper& helper) {
  TensorProto_DataType to;
  if (helper.HasSingleArgumentOfType<string>("to")) {
#ifndef CAFFE2_USE_LITE_PROTO
    string s = helper.GetSingleArgument<string>("to", "");
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    CAFFE_ENFORCE(
        TensorProto_DataType_Parse(s, &to), "Unknown 'to' argument: ", s);
#else
    CAFFE_THROW("String cast op not supported");
#endif
  } else {
    to = static_cast<TensorProto_DataType>(
        helper.GetSingleArgument<int>("to", TensorProto_DataType_UNDEFINED));
  }
  return to;
}
} // namespace cast

template <class Context>
class CastOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  CastOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    TensorProto_DataType to = cast::GetCastDataType(this->arg_helper());
    switch (to) {
      case TensorProto_DataType_FLOAT:
        body_ = &CastOp::DoRunWithDstType<float>;
        break;
      case TensorProto_DataType_INT32:
        body_ = &CastOp::DoRunWithDstType<int>;
        break;
      case TensorProto_DataType_BYTE:
        LOG(FATAL) << "BYTE is deprecated";
        break;
      case TensorProto_DataType_STRING:
        CAFFE_THROW("Casting to and from strings is not supported yet");
      // break;
      case TensorProto_DataType_BOOL:
        body_ = &CastOp::DoRunWithDstType<bool>;
        break;
      case TensorProto_DataType_UINT8:
        body_ = &CastOp::DoRunWithDstType<uint8_t>;
        break;
      case TensorProto_DataType_INT8:
        body_ = &CastOp::DoRunWithDstType<int8_t>;
        break;
      case TensorProto_DataType_UINT16:
        body_ = &CastOp::DoRunWithDstType<uint16_t>;
        break;
      case TensorProto_DataType_INT16:
        body_ = &CastOp::DoRunWithDstType<int16_t>;
        break;
      case TensorProto_DataType_INT64:
        body_ = &CastOp::DoRunWithDstType<int64_t>;
        break;
      case TensorProto_DataType_FLOAT16:
        CAFFE_THROW("Casting to and from float16 is not supported yet");
      // break;
      case TensorProto_DataType_DOUBLE:
        body_ = &CastOp::DoRunWithDstType<double>;
        break;
      case TensorProto_DataType_UNDEFINED:
        CAFFE_THROW("Cast op must have 'to' argument of type DataType");
      // break;
      default:
        CAFFE_THROW("Unexpected 'to' argument value: ", to);
    }
  }

  bool RunOnDevice() override {
    return (this->*body_)();
  }

  template <typename DstType>
  bool DoRunWithDstType() {
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

  template <typename DstType, typename SrcType>
  bool DoRunWithType() {
    auto& input = Input(0);
    auto* output = Output(0);
    output->ResizeLike(input);
    const auto* data = input.template data<SrcType>();
    auto* out = output->template mutable_data<DstType>();
    auto N = input.size();
    for (TIndex i = 0; i < N; ++i) {
      out[i] = static_cast<DstType>(data[i]);
    }
    return true;
  }

 private:
  bool (CastOp::*body_)();
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CAST_OP_H_
