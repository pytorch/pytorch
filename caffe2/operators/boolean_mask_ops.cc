#include "caffe2/operators/boolean_mask_ops.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {
namespace {

template <class Context>
class BooleanMaskLengthsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BooleanMaskLengthsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& lengths = Input(0);
    auto& mask = Input(1);
    auto* lengthsOut = Output(0);
    CAFFE_ENFORCE(lengths.ndim() == 1);
    CAFFE_ENFORCE(mask.ndim() == 1);
    const auto* lengthsPtr = lengths.template data<T>();
    const auto* maskPtr = mask.template data<bool>();
    auto totalLength =
        std::accumulate(lengthsPtr, lengthsPtr + lengths.size(), 0);
    CAFFE_ENFORCE(mask.size() == totalLength);
    lengthsOut->ResizeLike(lengths);
    auto* lengthsOutPtr = lengthsOut->template mutable_data<T>();
    int p = 0;
    for (int i = 0; i < lengths.size(); ++i) {
      T lengthOut = 0;
      for (int j = 0; j < lengthsPtr[i]; ++j) {
        if (maskPtr[p++]) {
          ++lengthOut;
        }
      }
      lengthsOutPtr[i] = lengthOut;
    }
    return true;
  }
};
} // namespace

template <>
bool BooleanMaskOp<CPUContext>::RunOnDevice() {
  auto& data = Input(0);
  auto& mask = Input(1);
  auto* dataOut = Output(0);
  CAFFE_ENFORCE(data.ndim() >= 1);
  CAFFE_ENFORCE_EQ(mask.ndim(), 1);
  CAFFE_ENFORCE(data.dims()[0] == mask.dims()[0]);

  const auto* maskPtr = mask.template data<bool>();
  int numOutputs = 0;
  int outerSize = mask.size();
  for (int i = 0; i < outerSize; ++i) {
    if (maskPtr[i]) {
      ++numOutputs;
    }
  }
  std::vector<TIndex> outShape;
  outShape.push_back(numOutputs);
  outShape.insert(outShape.end(), data.dims().begin() + 1, data.dims().end());
  dataOut->Resize(outShape);
  auto* outPtr = (char*)dataOut->raw_mutable_data(data.meta());

  int64_t* out_vec = nullptr;
  if (OutputSize() == 2) {
    auto* indicesOut = Output(1);
    indicesOut->Resize(numOutputs);
    out_vec = indicesOut->template mutable_data<int64_t>();
  }

  if (numOutputs == 0) {
    return true;
  }
  const auto innerSize = data.size_from_dim(1);
  const auto innerSizeBytes = innerSize * data.meta().itemsize();

  TIndex lastStart = -1;
  const auto* inPtr = (char*)data.raw_data();
  TIndex outStart = 0;

  for (TIndex i = 0;; ++i) {
    // mask was true and either a) became false, or b) sequence finished
    if (lastStart != -1 && ((i >= outerSize) || !maskPtr[i])) {
      const auto* src = inPtr + lastStart * innerSizeBytes;
      auto* dst = outPtr + outStart * innerSizeBytes;
      int numItems = i - lastStart;
      context_.template CopyItems<CPUContext, CPUContext>(
          data.meta(), numItems * innerSize, src, dst);
      outStart += numItems;
      lastStart = -1;
    }
    if (i >= outerSize) {
      break;
    }
    // mask was false and became true
    if (lastStart == -1 && maskPtr[i]) {
      lastStart = i;
    }
    if (maskPtr[i] && OutputSize() == 2) {
      *(out_vec++) = i;
    }
  }
  return true;
}

REGISTER_CPU_OPERATOR(BooleanMask, BooleanMaskOp<CPUContext>);
REGISTER_CPU_OPERATOR(BooleanMaskLengths, BooleanMaskLengthsOp<CPUContext>);

OPERATOR_SCHEMA(BooleanMask)
    .NumInputs(2)
    .NumOutputs(1, 2)
    .SetDoc(R"DOC(
Given a data tensor and a 1D boolean mask tensor, returns a tensor containing
only the elements corresponding to positions where the mask is true.
)DOC")
    .Input(0, "data", "The 1D, original data tensor.")
    .Input(1, "mask", "A tensor of bools of same shape as `data`.")
    .Output(0, "masked_data", "A tensor of same type as `data`.")
    .Output(1, "masked_indices", "A tensor for indices.");

OPERATOR_SCHEMA(BooleanMaskLengths)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given a tensor of int32 segment lengths and a mask (boolean) tensor, return
the segment lengths of a corresponding segmented tensor after BooleanMask is
applied.
)DOC")
    .Input(0, "lengths", "A 1D int32 tensor representing segment lengths.")
    .Input(1, "mask", "A 1D bool tensor of values to keep.")
    .Output(0, "masked_lengths", "Segment lengths of a masked tensor.");

NO_GRADIENT(BooleanMask)
NO_GRADIENT(BooleanMaskLengths);

const float minf = -1.0f * std::numeric_limits<float>::infinity();

// Template this on a functor object so we can generate different
// implementations at compile time and have a better chance of inlining
template <typename Functor>
void MaskWithFunctor(
    size_t N,
    size_t M,
    int B,
    const float* in,
    Functor fn,
    float fill_val,
    float* out) {
  if (B >= 0) { // with batching
    // collapse tensor to 3-dim view [B, N, M] where:
    // B is product of dims up to and including batch
    // N is product of dims between batch and axis, exclusive
    // M is product of dimensions at/after axis
    // then mask each batch [i, :, :] (note that this is N x M matrix)
    for (int i = 0; i < B; ++i) {
      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < M; ++k) {
          // when [i, :, :] is laid out in row major order
          // N * M * i + M * j + k is index of entry in N x M matrix
          // with coordinates (row = j, col = k)
          auto val = in[N * M * i + M * j + k];
          out[N * M * i + M * j + k] = (fn(j, k, val) ? fill_val : val);
        }
      }
    }
  } else { // without batching
    // TODO(T20952436): vector implementation
    // collapse tensor to 2-dim view [N, M], where
    // N is product of dimensions before axis
    // M is product of dimensions at/after axis
    // and mask N by M matrix
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        auto val = in[M * i + j];
        out[M * i + j] = (fn(i, j, val) ? fill_val : val);
      }
    }
  }
}

// Repeat masking along continuous segments (right axes) of size D
template <typename Functor>
void RepeatedMaskWithFunctor(
    size_t N,
    size_t M,
    int D,
    const float* in,
    Functor fn,
    float fill_val,
    float* out) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < D; ++k) {
        auto val = in[M * D * i + D * j + k];
        out[M * D * i + D * j + k] = (fn(i, j, val) ? fill_val : val);
      }
    }
  }
}

namespace {

class SequenceFunctor {
 public:
  explicit SequenceFunctor(const int* sl, const size_t len)
      : sl_(sl), len_(len) {}
  bool operator()(int i, int j, float /* val*/) {
    CAFFE_ENFORCE(i < len_, "Out of bound.");
    return j >= sl_[i];
  }

 private:
  const int* sl_;
  const size_t len_;
};

class WindowFunctor {
 public:
  explicit WindowFunctor(const int* c, int r) : c(c), r(r) {}
  bool operator()(int i, int j, float /* val*/) {
    return j > c[i] + r || j < c[i] - r;
  }

 private:
  const int* c;
  const int r;
};

class UpperFunctor {
 public:
  bool operator()(int i, int j, float /* val */) {
    return j > i;
  }
};

class LowerFunctor {
 public:
  bool operator()(int i, int j, float /* val */) {
    return j < i;
  }
};

class UpperDiagFunctor {
 public:
  bool operator()(int i, int j, float /* val */) {
    return j >= i;
  }
};

class LowerDiagFunctor {
 public:
  bool operator()(int i, int j, float /* val */) {
    return j <= i;
  }
};

} // namespace

template <>
bool SequenceMaskOp<CPUContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
}

template <>
template <class T>
bool SequenceMaskOp<CPUContext>::DoRunWithType() {
  const Tensor<CPUContext>* input = &Input(0);
  const Tensor<CPUContext>* sequence_lengths = nullptr;
  const Tensor<CPUContext>* window_centers = nullptr;

  if (mode_ == "sequence") {
    sequence_lengths = &Input(1);
  } else if (mode_ == "window") {
    window_centers = &Input(1);
  }

  auto* output = Output(0);
  output->ResizeLike(*input);

  const auto canonical_axis = input->canonical_axis_index(axis_);

  // canonical_batch is non-negative if batching, -1 otherwise
  int canonical_batch = -1;
  if ((HasArgument("batch"))) {
    canonical_batch = input->canonical_axis_index(batch_);
  }

  // make sure batch < axis
  if (canonical_batch >= 0) {
    CAFFE_ENFORCE_LT(canonical_batch, canonical_axis);
  }

  // if no batch, then left is product of dims up to axis
  // otherwise, left is product of dims between batch and axis
  const int left =
      (canonical_batch >= 0
           ? input->size_between_dim(canonical_batch, canonical_axis)
           : input->size_to_dim(canonical_axis));
  const int right = input->size_from_dim(canonical_axis);

  // product of dims from 1 to batch
  const int batch_dim =
      (canonical_batch >= 0
           ? input->size_to_dim(canonical_batch) * input->dim(canonical_batch)
           : -1);

  T fill_val = convert::To<float, T>(grad_ ? 0.0f : fill_val_);
  if (mode_ == "sequence") {
    CAFFE_ENFORCE(
        sequence_lengths, "Sequence length not provided for mode 'sequence'!");
    if (HasArgument("repeat_from_axis")) {
      const int canonical_repeat_from =
          input->canonical_axis_index(repeat_from_);
      const int repeated_dims = input->size_from_dim(canonical_repeat_from);
      const int masked_dims = right / repeated_dims;
      RepeatedMaskWithFunctor(
          left,
          masked_dims,
          repeated_dims,
          input->data<T>(),
          SequenceFunctor(
              sequence_lengths->data<int>(), sequence_lengths->size()),
          fill_val,
          output->mutable_data<T>());
    } else {
      MaskWithFunctor(
          left,
          right,
          batch_dim,
          input->data<T>(),
          SequenceFunctor(
              sequence_lengths->data<int>(), sequence_lengths->size()),
          fill_val,
          output->mutable_data<T>());
    }
  } else if (mode_ == "window") {
    MaskWithFunctor(
        left,
        right,
        batch_dim,
        input->data<T>(),
        WindowFunctor(window_centers->data<int>(), radius_),
        fill_val,
        output->mutable_data<T>());
  } else if (mode_ == "upper") {
    MaskWithFunctor(
        left,
        right,
        batch_dim,
        input->data<T>(),
        UpperFunctor(),
        fill_val,
        output->mutable_data<T>());
  } else if (mode_ == "lower") {
    MaskWithFunctor(
        left,
        right,
        batch_dim,
        input->data<T>(),
        LowerFunctor(),
        fill_val,
        output->mutable_data<T>());
  } else if (mode_ == "upperdiag") {
    MaskWithFunctor(
        left,
        right,
        batch_dim,
        input->data<T>(),
        UpperDiagFunctor(),
        fill_val,
        output->mutable_data<T>());
  } else if (mode_ == "lowerdiag") {
    MaskWithFunctor(
        left,
        right,
        batch_dim,
        input->data<T>(),
        LowerDiagFunctor(),
        fill_val,
        output->mutable_data<T>());
  } else {
    CAFFE_ENFORCE(false, "Unsupported mode for SequenceMaskOp!");
    return false;
  }

  return true;
}

REGISTER_CPU_OPERATOR(SequenceMask, SequenceMaskOp<CPUContext>);

OPERATOR_SCHEMA(SequenceMask)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Mask op designed for use in attention mechanisms for sequence modeling tasks.
Supports batching: given batch_dim, collapses dims 0 through batch_dim into a
single dimension, e.g. if tensor dims are [4,2,1,3,4] and batch_dim=2, first
collapse tensor to [4*2*1,3,4], then mask each batch [i,:,:].


Two current operating modes:


1) Given a 2D input tensor and 1D tensor of sequence lengths, for each row i in
the input tensor, set elements in that row to -inf if their column index
j >= sequence_lengths[i]. This mode takes two inputs and argument mode =
'sequence'


2) Triangular mask. Given row index i and column index j, set elements to -inf
given the following conditions:

      mode='upper', x_ij = -inf if j < i
      mode='lower', x_ij = -inf if j > i
      mode='upperdiag', x_ij = -inf if j <= i
      mode='lowerdiag', x_ij = -inf if j >= i

This mode takes one input.


3) Window Mask. Given a 2D input tensor and 1D tensor of window centers,
for each row i in the input tensor, set elements in that row to -inf
if their column index j outside [center - radius, center + radius].
This mode takes two inputs and argument mode = 'sequence'.
Argument 'radius' should be provided.
)DOC")
    .Input(0, "input", "Tensor to apply masking to")
    .Input(1, "sequence_lengths", "1D Tensor of sequence lengths for mode #1")
    .Output(0, "masked_tensor", "Input tensor with masking applied")
    .Arg(
        "mode",
        "(string) Mode selection. Possible values: "
        "'sequence', 'upper', 'lower', 'upperdiag', 'lowerdiag'")
    .Arg(
        "axis",
        "(int) Beginning axis of row elements. All dimensions to the left "
        "will be treated as row indices and those to the right (inclusive) "
        "will be treated as column indices in the 2D mask")
    .Arg("grad", "(bool) operate in gradient mode")
    .Arg("radius", "(int) radius of windows in window mode")
    .Arg("batch", "(int) batch dimension of tensor (optional)")
    .Arg(
        "repeat_from_axis",
        "(int) used when mask should be repeated for "
        "one or more data dimensions (beginning at this axis).  "
        "(currently only supported for sequence mode without batch argument)");

class GetSequenceMaskGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<Argument> args;
    args.reserve(Def().arg().size());
    for (const auto& x : Def().arg()) {
      args.push_back(x);
    }
    args.push_back(MakeArgument<bool>("grad", true));
    if (def_.input_size() == 1) {
      return SingleGradientDef(
          "SequenceMask",
          "",
          vector<string>{GO(0)},
          vector<string>{GI(0)},
          args);
    } else {
      return SingleGradientDef(
          "SequenceMask",
          "",
          vector<string>{GO(0), I(1)},
          vector<string>{GI(0)},
          args);
    }
  }

  bool CopyArguments() const override {
    return false;
  }
};

REGISTER_GRADIENT(SequenceMask, GetSequenceMaskGradient);

} // namespace caffe2
