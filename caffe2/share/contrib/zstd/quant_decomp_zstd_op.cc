#include "quant_decomp_zstd_op.h"
#include <stdint.h>
#include <zstd.h>
#include "caffe2/core/tensor.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {

namespace {

#define REGISTER_TYPE(index, type)                                      \
  {                                                                     \
    index, [](TensorCPU* tensor_) -> uint8_t* {                         \
      return reinterpret_cast<uint8_t*>(tensor_->mutable_data<type>()); \
    }                                                                   \
  }

// return a mutable pointer to the tensor in uint8_t format, the memory is
//   allocated based on the type 'type_index'
// supported type is defined in 'gTypeMapper'
uint8_t* GetMutableData(int type_index, TensorCPU* tensor) {
  // see COMP_DATA_TYPE_MAPPER in mutils.py for the mapping
  static const std::map<int, std::function<uint8_t*(TensorCPU * tensor)>>
      gTypeMapper = {REGISTER_TYPE(TensorProto::UINT8, uint8_t),
                     REGISTER_TYPE(TensorProto::UINT16, uint16_t),
                     REGISTER_TYPE(TensorProto::INT32, int32_t),
                     REGISTER_TYPE(TensorProto::FLOAT, float)};

  CAFFE_ENFORCE_EQ(
      gTypeMapper.count(type_index),
      1,
      "Invalid type index " + c10::to_string(type_index) + ".");
  return gTypeMapper.at(type_index)(tensor);
}

const uint8_t* GetCompressedPtr(const TensorCPU& compressed, size_t* out_size) {
  CAFFE_ENFORCE(
      // array of uint8_t
      compressed.template IsType<uint8_t>() ||
      // array with one string
      compressed.template IsType<std::string>());

  if (compressed.template IsType<uint8_t>()) {
    *out_size = compressed.numel();
    return compressed.data<uint8_t>();
  }

  // string type
  CAFFE_ENFORCE_EQ(compressed.numel(), 1);
  auto& str = compressed.data<std::string>()[0];
  *out_size = str.size();
  return reinterpret_cast<const uint8_t*>(str.data());
}

// Deserialize the string to get TensorProtos, storing tensors in compressed
// format
TensorProtos GetTensorsProto(const TensorCPU& compressed) {
  size_t sz;
  auto* ptr = GetCompressedPtr(compressed, &sz);
  TensorProtos tensors;
  CAFFE_ENFORCE(tensors.ParseFromArray(ptr, sz));
  return tensors;
}

// Decompress tensor stored in compressed format
// It is compressed using mutils.compress_data_list()
void Decompress(const TensorProto& compressed, TensorCPU* outDecomp) {
  vector<int64_t> shape(compressed.dims().begin(), compressed.dims().end());
  // shape stores the dimensions of data before compression,
  //   see _compress_data_single() in mutils.py
  outDecomp->Resize(shape);
  auto* out_ptr = GetMutableData(compressed.data_type(), outDecomp);

  auto* src = reinterpret_cast<const uint8_t*>(compressed.byte_data().data());
  size_t comp_size = compressed.byte_data().size();
  size_t decomp_size = outDecomp->nbytes();

  // call zstd
  size_t dc_size = ZSTD_decompress(out_ptr, decomp_size, src, comp_size);
  CAFFE_ENFORCE(!ZSTD_isError(dc_size), ZSTD_getErrorName(dc_size));
  CAFFE_ENFORCE_EQ(decomp_size, dc_size);
}

} // namespace

bool QuantDecompZstdOp::RunOnDevice() {
  const auto& op_compressed = Input(0);

  // Data could be an array of uint_t, or a string
  CAFFE_ENFORCE(
      // array of uint8_t
      op_compressed.template IsType<uint8_t>() ||
          // array with one string
          op_compressed.template IsType<std::string>(),
      op_compressed.dtype().name());

  // op_compressed: compressed data, 1d
  if (op_compressed.template IsType<uint8_t>()) {
    CAFFE_ENFORCE_EQ(op_compressed.dim(), 1, op_compressed.dim());
  } else {
    // string type has 0 dimension
    CAFFE_ENFORCE_EQ(op_compressed.numel(), 1, op_compressed.numel());
  }

  auto tensors = GetTensorsProto(op_compressed);
  CAFFE_ENFORCE_EQ(tensors.protos_size(), OutputSize());

  for (int i = 0; i < OutputSize(); i++) {
    Decompress(tensors.protos(i), Output(i));
  }

  return true;
}

REGISTER_CPU_OPERATOR(QuantDecompZstd, QuantDecompZstdOp);

OPERATOR_SCHEMA(QuantDecompZstd)
    .NumInputs(1)
    .NumOutputs(1, INT_MAX)
    .SetDoc(R"DOC(
 Decompress a set of tensors that are compressed using zstd.
 The data can be compressed using mutils.compress_data_list(), see
 quant_decomp_op_test.py for an example.
 The number of outputs depended on the input.
 )DOC")
    .Input(
        0,
        "compressed",
        "Compressed data in 1d tensor (uint8_t), "
        "or 0d tensor with one element in string type."
        "The data is compressed using mutils.compress_data_list().")
    .Output(0, "output0", "Decompressed data 0")
    .Output(1, "output1", "Decompressed data 1 if existed")
    .Output(2, "output2", "Decompressed data 2 if existed")
    .Output(3, "outputn", "Decompressed data n if existed");

SHOULD_NOT_DO_GRADIENT(QuantDecompZstd);

} // namespace caffe2
