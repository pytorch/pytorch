#include <ATen/core/op_registration/op_registration.h>
#include "caffe2/core/export_c10_op_to_caffe2.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

using caffe2::CPUContext;
using caffe2::Tensor;
using caffe2::TensorProto_DataType;

namespace caffe2 {
namespace {

template <typename DstType, typename SrcType>
void do_cast_(const Tensor& input, const Tensor& output) {
  output.ResizeLike(input);
  const auto* data = input.template data<SrcType>();
  auto* out = output.template mutable_data<DstType>();
  auto N = input.numel();
  for (int64_t i = 0; i < N; ++i) {
    out[i] = static_cast<DstType>(data[i]);
  }
}

template <class SrcType>
void cast_op_cpu_impl(
    const at::Tensor& input_,
    const at::Tensor& output_,
    int64_t to_) {
  Tensor input(input_);
  Tensor output(output_);
  TensorProto_DataType to = static_cast<TensorProto_DataType>(to_);

  switch (to) {
    case caffe2::TensorProto_DataType_FLOAT:
      do_cast_<float, SrcType>(input, output);
      break;
    case caffe2::TensorProto_DataType_INT32:
      do_cast_<int32_t, SrcType>(input, output);
      break;
    case caffe2::TensorProto_DataType_BYTE:
      LOG(FATAL) << "BYTE is deprecated";
      break;
    case caffe2::TensorProto_DataType_STRING:
      CAFFE_THROW("Casting to and from strings is not supported yet");
      // break;
    case caffe2::TensorProto_DataType_BOOL:
      do_cast_<bool, SrcType>(input, output);
      break;
    case caffe2::TensorProto_DataType_UINT8:
      do_cast_<uint8_t, SrcType>(input, output);
      break;
    case caffe2::TensorProto_DataType_INT8:
      do_cast_<int8_t, SrcType>(input, output);
      break;
    case caffe2::TensorProto_DataType_UINT16:
      do_cast_<uint16_t, SrcType>(input, output);
      break;
    case caffe2::TensorProto_DataType_INT16:
      do_cast_<int16_t, SrcType>(input, output);
      break;
    case caffe2::TensorProto_DataType_INT64:
      do_cast_<int64_t, SrcType>(input, output);
      break;
    case caffe2::TensorProto_DataType_FLOAT16:
      CAFFE_THROW("Casting to and from Half on CPU is not supported yet");
      // break;
    case caffe2::TensorProto_DataType_DOUBLE:
      do_cast_<double, SrcType>(input, output);
      break;
    case caffe2::TensorProto_DataType_UNDEFINED:
      CAFFE_THROW("Cast op must have 'to' argument of type DataType");
      // break;
    default:
      CAFFE_THROW("Unexpected 'to' argument value: ", to);
  }
}
void cast_op_cpu(
    const at::Tensor& input,
    const at::Tensor& output,
    int64_t to) {
  switch (input.scalar_type()) {
#define CASE(ctype,name) case ScalarType:: name : return cast_op_cpu_impl<ctype>(input, output, to);
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, CASE)
#undef CASE
    default: throw std::runtime_error(string() + "Unsupported scalar type " + toString(input.scalar_type()));
  }
}

static auto registry = c10::RegisterOperators().op(
    "_c10_experimental::Cast",
    c10::RegisterOperators::options()
      .kernel<decltype(cast_op_cpu), &cast_op_cpu>(DispatchKey::CPUTensorId));

} // namespace

C10_EXPORT_C10_OP_TO_CAFFE2_CPU(
    "_c10_experimental::Cast",
    C10Cast_DontUseThisOpYet)

} // namespace caffe2
