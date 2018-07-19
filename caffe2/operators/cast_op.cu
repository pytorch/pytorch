#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/cast_op.h"
#include "caffe2/utils/conversions.h"

namespace caffe2 {

template <typename DstType, typename SrcType>
__global__ void CastKernel(const int N, const SrcType* X, DstType* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    // Y[i] = static_cast<DstType>(X[i]);
    Y[i] = convert::To<SrcType, DstType>(X[i]);
  }
}

template <>
template <typename DstType, typename SrcType>
bool CastOp<CUDAContext>::DoRunWithType() {
  auto& input = Input(0);
  auto* output = Output(0);
  output->ResizeLike(input);
  const auto* data = input.template data<SrcType>();
  auto* out = output->template mutable_data<DstType>();
  DCHECK(input.size() < INT_MAX);
  int N = input.size();
  if (N == 0) {
    // skip the rest of the computation if input is empty
    return true;
  }
  CastKernel<DstType, SrcType>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(N, data, out);
  return true;
}

template <>
template <typename DstType>
bool CastOp<CUDAContext>::DoRunWithDstType() {
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

// specific version that allows for casting to fp16
template <>
template <>
bool CastOp<CUDAContext>::DoRunWithDstType<float>() {
  return DispatchHelper<
      TensorTypes<
          float,
          float16,
          int32_t,
          bool,
          uint8_t,
          int8_t,
          uint16_t,
          int16_t,
          int64_t,
          double>,
      float /* DstType */>::call(this, Input(0));
}

// specific version for casting _from_ fp16
template <>
template <>
bool CastOp<CUDAContext>::DoRunWithDstType<float16>() {
  return DispatchHelper<
      TensorTypes<
          float,
          float16>,
      float16 /* DstType */>::call(this, Input(0));
}
template <>
void CastOp<CUDAContext>::SetBody(TensorProto_DataType to) {
  switch (to) {
    case TensorProto_DataType_FLOAT:
      body_ = &CastOp<CUDAContext>::DoRunWithDstType<float>;
      break;
    case TensorProto_DataType_INT32:
      body_ = &CastOp<CUDAContext>::DoRunWithDstType<int>;
      break;
    case TensorProto_DataType_BYTE:
      LOG(FATAL) << "BYTE is deprecated";
      break;
    case TensorProto_DataType_STRING:
      CAFFE_THROW("Casting to and from strings is not supported yet");
      // break;
    case TensorProto_DataType_BOOL:
      body_ = &CastOp<CUDAContext>::DoRunWithDstType<bool>;
      break;
    case TensorProto_DataType_UINT8:
      body_ = &CastOp<CUDAContext>::DoRunWithDstType<uint8_t>;
      break;
    case TensorProto_DataType_INT8:
      body_ = &CastOp<CUDAContext>::DoRunWithDstType<int8_t>;
      break;
    case TensorProto_DataType_UINT16:
      body_ = &CastOp<CUDAContext>::DoRunWithDstType<uint16_t>;
      break;
    case TensorProto_DataType_INT16:
      body_ = &CastOp<CUDAContext>::DoRunWithDstType<int16_t>;
      break;
    case TensorProto_DataType_INT64:
      body_ = &CastOp<CUDAContext>::DoRunWithDstType<int64_t>;
      break;
    case TensorProto_DataType_FLOAT16:
      body_ = &CastOp<CUDAContext>::DoRunWithDstType<float16>;
      break;
    case TensorProto_DataType_DOUBLE:
      body_ = &CastOp<CUDAContext>::DoRunWithDstType<double>;
      break;
    case TensorProto_DataType_UNDEFINED:
      CAFFE_THROW("Cast op must have 'to' argument of type DataType");
      // break;
    default:
      CAFFE_THROW("Unexpected 'to' argument value: ", to);
  }
}

REGISTER_CUDA_OPERATOR(Cast, CastOp<CUDAContext>);

}  // namespace caffe2
