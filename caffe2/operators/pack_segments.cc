#include "caffe2/operators/pack_segments.h"

namespace caffe2 {

template <>
template <typename T>
bool PackSegmentsOp<CPUContext>::DoRunWithType() {
  return DispatchHelper<
      TensorTypes2<char, int32_t, int64_t, float, std::string>,
      T>::call(this, Input(DATA));
}

template <>
template <typename T, typename Data_T>
bool PackSegmentsOp<CPUContext>::DoRunWithType2() {
  const auto& data = Input(DATA);
  const auto& lengths = Input(LENGTHS);

  Tensor* presence_mask = nullptr;
  if (return_presence_mask_) {
    presence_mask = Output(1);
  }

  CAFFE_ENFORCE_GE(data.dim(), 1, "DATA should be at least 1-D");
  CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTH should be 1-D");

  // Find the length of the longest sequence.
  const T* l = lengths.template data<T>();
  T max_length = 0;
  int64_t total_length = 0;
  for (T i = 0; i < lengths.size(0); ++i) {
    max_length = std::max(max_length, l[i]);
    total_length += l[i];
  }
  if (max_length_ != -1) {
    max_length = max_length_;
  }

  // Total lengths must be the same as data.dims(0)
  CAFFE_ENFORCE_EQ(
      data.size(0),
      total_length,
      " PackSegments requires that the sum of the lengths ",
      total_length,
      " is equal to the first data dimension ",
      data.size(0));

  auto shape =
      data.sizes().vec(); // Shape of output is batch_size x max_len x ...
  shape[0] = max_length;
  shape.insert(shape.begin(), lengths.numel());
  auto* output = Output(0, shape, at::dtype(data.dtype()));

  // create output tensor
  auto* out = static_cast<char*>(output->raw_mutable_data(data.dtype()));

  bool* presence_mask_data = nullptr;
  if (return_presence_mask_) {
    // Shape of presence is batch_size x max_len
    std::vector<int64_t> presence_shape{lengths.numel(), max_length};
    presence_mask->Resize(presence_shape);
    presence_mask_data = presence_mask->template mutable_data<bool>();
  }

  if (!data.size(0)) {
    // Return empty output (with the proper shape)
    return true;
  }

  // Do padding
  // Ignore string since math::Set does not support string.
  // For all other cases, the behavior should mimic the GPU version where the
  // padding is always zero for types other than float.
  // TODO(xinyizhang): potentially restructure to clean up the logic here.
  if (output->template IsType<float>()) {
    math::Set<float, CPUContext>(
        output->numel(),
        padding_,
        output->template mutable_data<float>(),
        &context_);
  } else if (output->template IsType<int32_t>()) {
    math::Set<int32_t, CPUContext>(
        output->numel(),
        0,
        output->template mutable_data<int32_t>(),
        &context_);
  } else if (output->template IsType<int64_t>()) {
    math::Set<int64_t, CPUContext>(
        output->numel(),
        0,
        output->template mutable_data<int64_t>(),
        &context_);
  } else if (output->template IsType<char>()) {
    math::Set<char, CPUContext>(
        output->numel(), 0, output->template mutable_data<char>(), &context_);
  }
  if (return_presence_mask_) {
    memset(presence_mask_data, (int)false, presence_mask->numel());
  }

  auto block_size = data.size_from_dim(1);
  auto block_bytesize = data.itemsize() * block_size;
  const auto* d = static_cast<const char*>(data.raw_data());
  int64_t start = 0;
  for (int64_t i = 0; i < lengths.size(0); ++i) {
    auto len = l[i] <= max_length ? l[i] : max_length;
    context_.CopyItemsSameDevice(
        data.dtype(),
        len * block_size,
        d + block_bytesize * start,
        out + block_bytesize * max_length * i);
    if (return_presence_mask_) {
      memset(presence_mask_data + max_length * i, (int)true, len);
    }
    start += l[i];
  }

  return true;
}

template <>
template <typename T>
bool UnpackSegmentsOp<CPUContext>::DoRunWithType() {
  return DispatchHelper<
      TensorTypes2<char, int32_t, int64_t, float, std::string>,
      T>::call(this, Input(DATA));
}

template <>
template <typename T, typename Data_T>
bool UnpackSegmentsOp<CPUContext>::DoRunWithType2() {
  const auto& data = Input(DATA);
  const auto& lengths = Input(LENGTHS);
  auto* output = Output(0);

  CAFFE_ENFORCE_GE(data.dim(), 2, "DATA should be at least 2-D");
  CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTH should be 1-D");
  if (max_length_ != -1) {
    CAFFE_ENFORCE_EQ(
        max_length_,
        data.size(1),
        "max_length should be equal to the second dimension of the packed segments");
  }
  const T* l = lengths.template data<T>();

  int64_t total_l = 0;
  if (max_length_ != -1) {
    for (int64_t i = 0; i < lengths.size(0); ++i) {
      total_l += (int64_t)(l[i] <= max_length_ ? l[i] : max_length_);
    }
  } else {
    total_l = std::accumulate(l, l + lengths.size(0), (int64_t)0);
  }

  auto shape = data.sizes().vec();
  CAFFE_ENFORCE_EQ(
      shape[0], lengths.size(0), "LENGTH should match DATA in dimension 0");
  shape.erase(shape.begin());
  shape[0] = total_l;
  output->Resize(shape);
  // create output tensor
  auto* out = static_cast<char*>(output->raw_mutable_data(data.dtype()));
  if (!(data.size(0) && data.size(1))) {
    return true;
  }
  auto block_size = data.size_from_dim(2);
  auto block_bytesize = data.itemsize() * block_size;
  const auto* d = static_cast<const char*>(data.raw_data());
  int64_t start = 0;
  for (int64_t i = 0; i < lengths.size(0); ++i) {
    auto len = l[i];
    if (max_length_ != -1 && l[i] > max_length_) {
      len = max_length_;
    }
    context_.CopyItemsSameDevice(
        data.dtype(),
        len * block_size,
        d + block_bytesize * data.size(1) * i,
        out + block_bytesize * start);
    start += len;
  }
  return true;
}

REGISTER_CPU_OPERATOR(PackSegments, PackSegmentsOp<CPUContext>);
REGISTER_CPU_OPERATOR(UnpackSegments, UnpackSegmentsOp<CPUContext>);

OPERATOR_SCHEMA(PackSegments)
    .NumInputs(2)
    .NumOutputs(1, 2)
    .SetDoc(
        "Map N dim tensor to N+1 dim based on length blob. Sequences that \
    are shorter than the longest sequence are padded with zeros.")
    .Input(
        0,
        "lengths",
        "1-d int/long tensor contains the length in each of the output.")
    .Input(1, "tensor", "N dim Tensor.")
    .Output(
        0,
        "packed_tensor",
        "N + 1 dim Tensor"
        "where dim(1) is the max length"
        ", dim(0) is the batch size.")
    .Output(
        1,
        "presence_mask",
        "2 dim boolean tensor"
        ", false where packed_tensor is padded, true otherwise.")
    .Arg("max_length", "The pre-defined max_length for the packed segments")
    .Arg(
        "pad_minf",
        "Padding number in the packed segments. Use true to pad \
    -infinity, otherwise pad zeros")
    .Arg(
        "return_presence_mask",
        "bool whether to return presence mask, false by default");
OPERATOR_SCHEMA(UnpackSegments)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc("Map N+1 dim tensor to N dim based on length blob")
    .Input(
        0,
        "lengths",
        "1-d int/long tensor contains the length in each of the input.")
    .Input(1, "tensor", "N+1 dim Tensor.")
    .Output(0, "packed_tensor", "N dim Tensor")
    .Arg("max_length", "The pre-defined max_length for the packed segments");

class GetPackSegmentsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "UnpackSegments",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(1)});
  }
};
REGISTER_GRADIENT(PackSegments, GetPackSegmentsGradient);

class GetUnpackSegmentsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "PackSegments", "", vector<string>{I(0), GO(0)}, vector<string>{GI(1)});
  }
};
REGISTER_GRADIENT(UnpackSegments, GetUnpackSegmentsGradient);
} // namespace caffe2
