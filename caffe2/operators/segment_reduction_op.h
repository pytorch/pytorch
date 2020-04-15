#ifndef CAFFE2_OPERATORS_SEGMENT_REDUCTION_OP_H_
#define CAFFE2_OPERATORS_SEGMENT_REDUCTION_OP_H_

#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/reducer_functors.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(LengthsSum);
C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(LengthsMean);
C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(LengthsMax);

namespace caffe2 {

template <typename TData>
class BaseInputAccessor {
 public:
  BaseInputAccessor() {}

  bool observeInput(const Tensor& dataInput) {
    data_ = dataInput.raw_data();
    return dataInput.template IsType<TData>();
  }

  inline const TData*
  getBlockPtr(int64_t in_block_size, int64_t idx, int64_t /* blocks */ = 1) {
    return static_cast<const TData*>(data_) + in_block_size * idx;
  }

 protected:
  const void* data_ = nullptr;
};

////////////////////////////////////////////////////////////////////////////////
// Range reducer ops: leverage that input segment is continuous and allow
// reducer functors to do something special
// Note: for now there are no real use cases for it yet :)
// Also, doesn't support additional arguments for now
////////////////////////////////////////////////////////////////////////////////

/**
 * Base implementation for segment reduction op that leverages continuity of the
 * data
 *
 * Assumes that segments are sorted and there are no skip indices
 */
template <
    typename T,
    typename SIndex,
    class Context,
    class RangeReducer,
    class InputAccessor = BaseInputAccessor<T>>
class AbstractSortedSegmentRangeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AbstractSortedSegmentRangeOp);

  bool RunOnDevice() override {
    auto& dataInput = Input(DATA);
    auto& segment_ids = Input(SEGMENT_IDS);

    CAFFE_ENFORCE_EQ(1, segment_ids.dim(), "SEGMENT_IDS must be a vector");
    auto N = segment_ids.size(0);
    CAFFE_ENFORCE_EQ(
        N,
        dataInput.size(0),
        "SEGMENT_IDS must have the same length as outer dimension of DATA");

    OPERATOR_NEEDS_FEATURE(
        inputAccessor_.observeInput(dataInput),
        "Unsupported input type: ",
        dataInput.dtype().name(),
        ".");

    const SIndex* s_ids = segment_ids.template data<SIndex>();

    const SIndex K = N > 0 ? s_ids[N - 1] + 1 : 0;
    auto shape = dataInput.sizes().vec();
    shape[0] = K;
    auto* output = Output(0, shape, at::dtype<T>());

    T* out = output->template mutable_data<T>();

    if (N == 0) {
      return true;
    }

    int64_t block_size = dataInput.numel() / N;

    // Assume the segments are sorted and there are no gaps
    CAFFE_ENFORCE_EQ(0, s_ids[0], "Indices must be sorted and not have gaps");
    for (int64_t i = 0; i < N;) {
      int64_t start = i;
      for (++i; i < N && s_ids[start] == s_ids[i]; ++i)
        ;

      RangeReducer()(
          block_size,
          i - start,
          inputAccessor_.getBlockPtr(block_size, start, i - start),
          out + block_size * s_ids[start],
          &context_);

      // check correctness of the next segment
      if (i < N) {
        CAFFE_ENFORCE_EQ(
            s_ids[start] + 1,
            s_ids[i],
            "Indices must be sorted and not have gaps");
      }
    }
    return true;
  }

  static constexpr int kNumInputs = 2;
  INPUT_TAGS(DATA, SEGMENT_IDS);

 private:
  InputAccessor inputAccessor_;
};

template <
    typename T,
    typename SIndex,
    class Context,
    class RangeReducerGradient>
class AbstractSortedSegmentRangeGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AbstractSortedSegmentRangeGradientOp);

  bool RunOnDevice() override {
    // TODO(azzolini): avoid using input/output if not used by a particular op
    auto& data_in = Input(DATA_IN);
    auto& data_out = Input(DATA_OUT);
    auto& segment_grads = Input(SEGMENT_GRADS);
    auto& segment_ids = Input(SEGMENT_IDS);

    CAFFE_ENFORCE_EQ(1, segment_ids.dim(), "SEGMENT_IDS must be a vector");
    int64_t N = segment_ids.size(0);

    const SIndex* s_ids = segment_ids.template data<SIndex>();
    const T* s_grads = segment_grads.template data<T>();
    const T* d_in = data_in.template data<T>();
    const T* d_out = data_out.template data<T>();

    auto shape = segment_grads.sizes().vec();
    shape[0] = N;
    auto* data_grads = Output(0, shape, at::dtype<T>());

    const SIndex K = segment_grads.size(0);
    T* out = data_grads->template mutable_data<T>();

    if (N == 0) {
      return true;
    }

    int64_t block_size = segment_grads.size_from_dim(1);

    // Assume the segments are sorted and there are no gaps
    CAFFE_ENFORCE_EQ(0, s_ids[0], "Indices must be sorted and not have gaps");
    // repeat the check from forward op
    CAFFE_ENFORCE_EQ(
        K - 1, s_ids[N - 1], "Indices must be sorted and not have gaps");
    for (int64_t i = 0; i < N;) {
      int64_t start = i;
      for (++i; i < N && s_ids[start] == s_ids[i]; ++i)
        ;

      auto expanded_idx = block_size * start;
      auto reduced_idx = block_size * s_ids[start];
      RangeReducerGradient()(
          block_size,
          i - start,
          s_grads + reduced_idx,
          out + expanded_idx,
          d_in + expanded_idx,
          d_out + reduced_idx,
          &context_);

      // check correctness of the next segment
      if (i < N) {
        CAFFE_ENFORCE_EQ(
            s_ids[start] + 1,
            s_ids[i],
            "Indices must be sorted and not have gaps");
      }
    }
    return true;
  }

  static constexpr int kNumInputs = 4;
  INPUT_TAGS(DATA_IN, DATA_OUT, SEGMENT_GRADS, SEGMENT_IDS);
};

template <typename T, typename SIndex, typename Context, typename ReducerDef>
struct AbstractSortedSegmentRangeDef {
  using OpDef = ReducerDef;
  static constexpr const char* basename = "SortedSegmentRange";
  static constexpr const char* doc = R"DOC(
Applies '{op}' to each segment of input tensor. In order to allow for more
efficient implementation of '{op}', the input segments have to be contiguous
and non-empty.

SEGMENT_IDS is a vector that maps each of the first dimension slices of the
DATA to a particular group (segment). Values belonging to the same segment are
aggregated together.

The first dimension of the output is equal to the number of input segments,
i.e. `SEGMENT_IDS[-1]+1`. Other dimensions are inherited from the input tensor.

{op_doc}
  )DOC";
  static void PopulateSchema(OpSchema& schema) {
    schema.Input(0, "DATA", "Input tensor to be aggregated");
    schema.Input(
        1,
        "SEGMENT_IDS",
        "Vector with the same length as the first dimension of DATA "
        "and values in the range 0..K-1 and in increasing order that "
        "maps each slice of DATA to one of the segments");
    schema.Output(
        0,
        "OUTPUT",
        "Aggregated tensor with the first dimension of K and the "
        "other dimentsions inherited from DATA");
  }
  using ForwardOp = AbstractSortedSegmentRangeOp<
      T,
      SIndex,
      Context,
      typename ReducerDef::template Reducer<T, Context>>;
  using BackwardOp = AbstractSortedSegmentRangeGradientOp<
      T,
      SIndex,
      Context,
      typename ReducerDef::template ReducerGradient<T, Context>>;
  struct GetGradient : public GradientMakerBase {
    using GradientMakerBase::GradientMakerBase;
    vector<OperatorDef> GetGradientDefs() override {
      return SingleGradientDef(
          string(basename) + ReducerDef::name + "Gradient",
          "",
          vector<string>{I(0), O(0), GO(0), I(1)},
          // no gradient on segment_ids!
          vector<string>{GI(0)});
    }
  };
};

////////////////////////////////////////////////////////////////////////////////
// Incremental reducer ops: assume that reducer consumes pieces of data one by
// one. Also, supports additional arguments passed to reducer, e.g. scalers for
// weighted sum.
//
// Note: in current implementation additional inputs are considered auxiliary
// constants and have limitations:
// - there is no gradient computation for auxiliary inputs
// - auxiliary inputs aren't affected by fused embedding lookup in operations
// like sparse_sorted_segment
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Simple non-segmented reduction over the first few dimensions of the
 * tensor
 *
 * Inputs:
 *   0: DATA - input embedding to do lookups in
 *   1..P: AUX_ARG_<I> - optional additional arguments to be passed to the
 *                       reducer
 *
 * Args:
 *   num_reduce_dim (default 1) - the number of dims in front of the tensor to
 *                                reduce
 *
 * Output:
 *   Tensor without the first `num_dim` dimensions of DATA
 */
template <
    typename T,
    class Context,
    class Reducer,
    bool FirstDim,
    class InputAccessor = BaseInputAccessor<T>>
class AbstractReduceFrontOrBackOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit AbstractReduceFrontOrBackOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "num_reduce_dim", num_reduce_dims_, 1) {}

  bool RunOnDevice() override {
    auto& data = Input(0);
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    int64_t in_block_size = FirstDim
        ? data.size_from_dim(num_reduce_dims_)
        : data.size_to_dim(data.dim() - num_reduce_dims_);
    return DispatchHelper<typename Reducer::FixedDispatch>::call(
        this, in_block_size);
  }

  template <int FixedSize>
  bool DoRunWithValue() {
    auto& data = Input(0);

    CAFFE_ENFORCE_LE(num_reduce_dims_, data.dim());

    typename Reducer::Meta ctx(FirstDim);
    ctx.observeInput(0, data, num_reduce_dims_);
    for (int i = 1; i < Reducer::kInputCount; ++i) {
      auto& aux_in = Input(i);
      ctx.observeInput(i, aux_in, num_reduce_dims_);
    }

    OPERATOR_NEEDS_FEATURE(
        inputAccessor_.observeInput(data),
        "Unsupported input type: ",
        data.dtype().name(),
        ".");

    vector<int64_t> shape;
    ctx.appendOutputShape(&shape);
    auto* output = Output(0, shape, at::dtype<T>());

    T* out = output->template mutable_data<T>();

    const int block_size = FirstDim
        ? data.size_from_dim(num_reduce_dims_)
        : data.size_from_dim(data.dim() - num_reduce_dims_);

    const int num_blocks = block_size > 0 ? data.numel() / block_size : 0;

    Reducer r(ctx, out, &context_);
    for (int64_t i = 0; i < num_blocks; ++i) {
      r.template process<FixedSize>(
          ctx, inputAccessor_.getBlockPtr(block_size, i), i, &context_);
    }
    r.template finish<FixedSize>(ctx, &context_);
    return true;
  }

  static constexpr int kNumInputs = Reducer::kInputCount;

 private:
  int num_reduce_dims_;
  InputAccessor inputAccessor_;
};

template <
    typename T,
    class Context,
    class ReducerGradient,
    bool FirstDim = true>
class AbstractReduceFrontOrBackGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit AbstractReduceFrontOrBackGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "num_reduce_dim", num_reduce_dims_, 1) {}

  bool RunOnDevice() override {
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    int64_t grad_block_size = Input(REDUCTION_GRAD).numel();
    return DispatchHelper<typename ReducerGradient::FixedDispatch>::call(
        this, grad_block_size);
  }

  template <int FixedSize>
  bool DoRunWithValue() {
    auto& reduction_grad = Input(REDUCTION_GRAD);
    auto& source_shape = this->template Input<Tensor>(SOURCE_SHAPE, CPU);

    typename ReducerGradient::Meta ctx(reduction_grad, 0, FirstDim);
    for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
      auto& aux_in = Input(i);
      ctx.observeOriginalInput(
          ReducerGradient::originalInputs()[i],
          aux_in,
          nullptr, /*no grad*/
          num_reduce_dims_);
    }

    const T* r_grad = reduction_grad.template data<T>();

    CAFFE_ENFORCE_LE(num_reduce_dims_, source_shape.numel());

    vector<int64_t> shape(
        source_shape.template data<int64_t>(),
        source_shape.template data<int64_t>() + source_shape.numel());

    auto* data_grads = Output(0, shape, at::dtype<T>());

    int64_t block_size = FirstDim
        ? data_grads->size_from_dim(num_reduce_dims_)
        : data_grads->size_from_dim(data_grads->dim() - num_reduce_dims_);
    int64_t block_num = block_size > 0 ? data_grads->numel() / block_size : 0;

    T* out = data_grads->template mutable_data<T>();

    ReducerGradient r(ctx, r_grad, &context_);
    for (int64_t i = 0; i < block_num; ++i) {
      r.template fillGrad<FixedSize>(
          ctx,
          out + block_size * i,
          i,
          &context_,
          FirstDim ? block_num : block_size);
    }
    return true;
  }

  static constexpr int kNumInputs =
      ReducerGradient::originalInputs().size() + 2;
  enum _InputTags {
    REDUCTION_GRAD = ReducerGradient::originalInputs().size(),
    SOURCE_SHAPE
  };

 private:
  int num_reduce_dims_;
};

template <typename T, typename Context, typename ReducerDef>
struct AbstractReduceFrontDef {
  using OpDef = ReducerDef;
  static constexpr const char* basename = "ReduceFront";
  static constexpr const char* doc = R"DOC(
Reduces the input tensor along the first dimension of the input tensor by
applying '{op}'. This op acts in a similar way to SortedSegment{op} and
UnsortedSegment{op} but as if all input slices belong to a single segment.

{op_doc}
  )DOC";
  static void PopulateSchema(OpSchema& schema) {
    schema.Input(
        0, "DATA", "Input tensor to be reduced on the first dimension");
    schema.TensorInferenceFunction([](const OperatorDef& def,
                                      const vector<TensorShape>& in) {
      CAFFE_ENFORCE_EQ(1, in.size());
      ArgumentHelper helper(def);
      int num_reduce_dims = helper.GetSingleArgument<int>("num_reduce_dim", 1);
      typename ReducerDef::template Reducer<T, Context>::Meta ctx(true);
      vector<int64_t> out_dims = ctx.getOutputShape(in[0], num_reduce_dims);
      return vector<TensorShape>{
          CreateTensorShape(out_dims, in[0].data_type())};
    });
    ReducerDef::PopulateSchema(schema);
  }
  using ReducerGradient =
      typename ReducerDef::template ReducerGradient<T, Context>;
  using ForwardOp = AbstractReduceFrontOrBackOp<
      T,
      Context,
      typename ReducerDef::template Reducer<T, Context>,
      true>;
  using BackwardOp =
      AbstractReduceFrontOrBackGradientOp<T, Context, ReducerGradient, true>;
  struct GetGradient : public GradientMakerBase {
    using GradientMakerBase::GradientMakerBase;
    vector<OperatorDef> GetGradientDefs() override {
      // Have utility function generating these names?
      string tmp_dims = "_" + O(0) + "_dims";

      vector<string> grad_ins;
      for (const int i : ReducerGradient::originalInputs()) {
        grad_ins.push_back(I(i));
      }
      grad_ins.push_back(GO(0));
      grad_ins.push_back(tmp_dims);

      vector<Argument> args;
      if (ArgumentHelper::HasArgument(def_, "num_reduce_dim")) {
        args.push_back(GetArgument(def_, "num_reduce_dim"));
      }
      // FIXME: pass in num_reduce_dims?!
      return vector<OperatorDef>{
          CreateOperatorDef(
              "Shape", "", vector<string>{I(0)}, vector<string>{tmp_dims}),
          CreateOperatorDef(
              string(basename) + ReducerDef::name + "Gradient",
              "",
              grad_ins,
              // no gradient on auxiliary inputs for now
              vector<string>{GI(0)}),
      };
    }
  };
};

template <typename T, typename Context, typename ReducerDef>
struct AbstractReduceBackDef {
  using OpDef = ReducerDef;
  static constexpr const char* basename = "ReduceBack";
  static constexpr const char* doc = R"DOC(
Reduces the input tensor along the last dimension of the input tensor by
applying '{op}'. This op acts in a similar way to SortedSegment{op} and
UnsortedSegment{op} but as if all input slices belong to a single segment.

{op_doc}
  )DOC";
  static void PopulateSchema(OpSchema& schema) {
    schema.Input(
        0, "DATA", "Input tensor to be reduced on the first dimension");
    schema.TensorInferenceFunction([](const OperatorDef& def,
                                      const vector<TensorShape>& in) {
      CAFFE_ENFORCE_EQ(1, in.size());
      ArgumentHelper helper(def);
      int num_reduce_dims = helper.GetSingleArgument<int>("num_reduce_dim", 1);
      typename ReducerDef::template Reducer<T, Context>::Meta ctx(false);
      vector<int64_t> out_dims = ctx.getOutputShape(in[0], num_reduce_dims);
      return vector<TensorShape>{
          CreateTensorShape(out_dims, in[0].data_type())};
    });
    ReducerDef::PopulateSchema(schema);
  }
  using ReducerGradient =
      typename ReducerDef::template ReducerGradient<T, Context>;
  using ForwardOp = AbstractReduceFrontOrBackOp<
      T,
      Context,
      typename ReducerDef::template Reducer<T, Context>,
      false>;
  using BackwardOp =
      AbstractReduceFrontOrBackGradientOp<T, Context, ReducerGradient, false>;
  struct GetGradient : public GradientMakerBase {
    using GradientMakerBase::GradientMakerBase;
    vector<OperatorDef> GetGradientDefs() override {
      // Have utility function generating these names?
      string tmp_dims = "_" + O(0) + "_dims";

      vector<string> grad_ins;
      for (const int i : ReducerGradient::originalInputs()) {
        grad_ins.push_back(I(i));
      }
      grad_ins.push_back(GO(0));
      grad_ins.push_back(tmp_dims);

      vector<Argument> args;
      if (ArgumentHelper::HasArgument(def_, "num_reduce_dim")) {
        args.push_back(GetArgument(def_, "num_reduce_dim"));
      }
      // FIXME: pass in num_reduce_dims?!
      return vector<OperatorDef>{
          CreateOperatorDef(
              "Shape", "", vector<string>{I(0)}, vector<string>{tmp_dims}),
          CreateOperatorDef(
              string(basename) + ReducerDef::name + "Gradient",
              "",
              grad_ins,
              // no gradient on auxiliary inputs for now
              vector<string>{GI(0)}),
      };
    }
  };
};

/**
 * @brief Segment reduction op with optional fused embedding lookup
 *
 * Base implementation for SortedSegmentXXX and SparseSortedSegmentXXX depending
 * on SparseFused static argument.
 *
 * Inputs:
 *   0: DATA - input embedding to do lookups in
 *   1..P: AUX_ARG_<I> - optional additional arguments to be passed to the
 *                       reducer, should have the same first dimension as
 *                       SEGMENT_IDS (e.g. scalars in WeightedSum)
 *   # if SparseFused == true:
 *   P+1: INDICES - 1-D vector with indices to look up in DATA. Should have the
 *                  same dimension as SEGMENT_IDS
 *   # P+1 if SparseFused == false:
 *   P+1 or P+2: SEGMENT_IDS - sorted segment ids 1-D vector
 *
 * Output:
 *   Tensor with first dimension of K, where K is the max segment id + 1. Rest
 *   of dimensions are decided by reducer but usually are the same size as extra
 *   dimensions of DATA
 */
template <
    typename T,
    typename SIndex,
    class Context,
    class Reducer,
    bool SparseFused = true,
    class InputAccessor = BaseInputAccessor<T>>
class AbstractSortedSegmentOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AbstractSortedSegmentOp);

  bool RunOnDevice() override {
    if (SparseFused) {
      return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
          this, Input(INDICES));
    } else {
      // type doesn't matter
      return DoRunWithType<int64_t>();
    }
  }

  template <typename IndexType>
  bool DoRunWithType() {
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    int64_t in_block_size = Input(0).size_from_dim(1);
    return DispatchHelper<typename Reducer::FixedDispatch, IndexType>::call(
        this, in_block_size);
  }

  template <typename IndexType, int FixedSize>
  bool DoRunWithValue() {
    auto& dataInput = Input(0);
    auto& segment_ids = Input(SEGMENT_IDS);

    CAFFE_ENFORCE_EQ(1, segment_ids.dim(), "SEGMENT_IDS must be a vector");
    int64_t N = segment_ids.size(0);
    const int64_t M = dataInput.size(0);

    const IndexType* idxs;
    if (SparseFused) { // static if
      auto& indices = Input(INDICES);
      CAFFE_ENFORCE_EQ(1, indices.dim(), "INDICES must be a vector");
      CAFFE_ENFORCE_EQ(
          N,
          indices.size(0),
          "SEGMENT_IDS must have the same length as INDICES");
      idxs = indices.template data<IndexType>();
    } else {
      CAFFE_ENFORCE_EQ(
          N, M, "DATA must have the same first dimension as SEGMENT_IDS");
    }

    // It would probably look nicer with varargs templates but it's too much
    // metaprogramming
    typename Reducer::Meta ctx;
    ctx.observeInput(0, dataInput, 1);
    for (int i = 1; i < Reducer::kInputCount; ++i) {
      auto& aux_in = Input(i);
      CAFFE_ENFORCE_EQ(
          N,
          aux_in.size(0),
          "Input ",
          i,
          " must have the same first dim as SEGMENT_IDS");
      ctx.observeInput(i, aux_in, 1);
    }

    OPERATOR_NEEDS_FEATURE(
        inputAccessor_.observeInput(dataInput),
        "Unsupported input type: ",
        dataInput.dtype().name(),
        ".");

    const SIndex* s_ids = segment_ids.template data<SIndex>();

    const SIndex K = N > 0 ? s_ids[N - 1] + 1 : 0;
    vector<int64_t> shape;
    shape.push_back(K);
    ctx.appendOutputShape(&shape);
    auto* output = Output(0, shape, at::dtype<T>());

    T* out = output->template mutable_data<T>();
    if (N == 0) {
      return true;
    }
    int64_t in_block_size = dataInput.size_from_dim(1);
    int64_t out_block_size = output->size_from_dim(1);

    // Assume the segments are sorted and there are no gaps
    CAFFE_ENFORCE_EQ(0, s_ids[0], "Indices must be sorted and not have gaps");
    for (int64_t i = 0; i < N;) {
      int64_t start = i;

      Reducer r(ctx, out + out_block_size * s_ids[start], &context_);
      for (; i < N && s_ids[start] == s_ids[i]; ++i) {
        IndexType idx;
        if (SparseFused) { // static if
          CAFFE_ENFORCE(
              0 <= idxs[i] && idxs[i] < M,
              "Index out of bounds: ",
              idxs[i],
              ", range 0 to ",
              M);
          idx = idxs[i];
        } else {
          idx = i;
        }
        r.template process<FixedSize>(
            ctx, inputAccessor_.getBlockPtr(in_block_size, idx), i, &context_);
      }

      r.template finish<FixedSize>(ctx, &context_);
      // check correctness of the next segment
      if (i < N) {
        CAFFE_ENFORCE_EQ(
            s_ids[start] + 1,
            s_ids[i],
            "Indices must be sorted and not have gaps");
      }
    }
    return true;
  }

  enum {
    INDICES = Reducer::kInputCount,
    SEGMENT_IDS = Reducer::kInputCount + (SparseFused ? 1 : 0)
  };
  static constexpr int kSelfInputs = SparseFused ? 2 : 1;
  static constexpr int kNumInputs = Reducer::kInputCount + kSelfInputs;

 private:
  InputAccessor inputAccessor_;
};

// Gradient actually doesn't depend on whether sparse lookup is fused or not
template <typename T, typename SIndex, class Context, class ReducerGradient>
class AbstractSortedSegmentGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AbstractSortedSegmentGradientOp);

  bool RunOnDevice() override {
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    int64_t grad_block_size = Input(SEGMENT_GRADS).size_from_dim(1);
    return DispatchHelper<typename ReducerGradient::FixedDispatch>::call(
        this, grad_block_size);
  }

  template <int FixedSize>
  bool DoRunWithValue() {
    auto& segment_grads = Input(SEGMENT_GRADS);
    auto& segment_ids = Input(SEGMENT_IDS);

    CAFFE_ENFORCE_EQ(1, segment_ids.dim(), "SEGMENT_IDS must be a vector");
    int64_t N = segment_ids.size(0);

    typename ReducerGradient::Meta ctx(segment_grads, 1);
    for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
      auto& aux_in = Input(i);
      CAFFE_ENFORCE_EQ(
          N,
          aux_in.size(0),
          "Input ",
          i,
          " must have the same first dim as SEGMENT_IDS");
      ctx.observeOriginalInput(
          ReducerGradient::originalInputs()[i], aux_in, nullptr /*no grad*/, 1);
    }

    const SIndex* s_ids = segment_ids.template data<SIndex>();
    const T* s_grads = segment_grads.template data<T>();

    vector<int64_t> shape;
    shape.push_back(N);
    ctx.appendGradShape(&shape);
    auto* data_grads = Output(0, shape, at::dtype<T>());

    int64_t d_block_size = data_grads->size_from_dim(1);
    const SIndex K = segment_grads.size(0);
    int64_t s_block_size = segment_grads.size_from_dim(1);
    T* out = data_grads->template mutable_data<T>();

    if (N == 0) {
      return true;
    }

    // Assume the segments are sorted and there are no gaps
    CAFFE_ENFORCE_EQ(0, s_ids[0], "Indices must be sorted and not have gaps");
    // repeat the check from forward op
    CAFFE_ENFORCE_EQ(
        K - 1, s_ids[N - 1], "Indices must be sorted and not have gaps");
    for (int64_t i = 0; i < N;) {
      int64_t start = i;
      int64_t end = start;

      if (ReducerGradient::computeLength()) {
        for (; end < N && s_ids[start] == s_ids[end]; ++end) {
        }
      }

      ReducerGradient r(ctx, s_grads + s_block_size * s_ids[start], &context_);
      for (; i < N && s_ids[start] == s_ids[i]; ++i) {
        r.template fillGrad<FixedSize>(
            ctx, out + d_block_size * i, i, &context_, end - start);
      }

      // check correctness of the next segment
      if (i < N) {
        CAFFE_ENFORCE_EQ(
            s_ids[start] + 1,
            s_ids[i],
            "Indices must be sorted and not have gaps");
      }
    }
    return true;
  }

  // Input layout:
  //   orig_arg1, orig_arg2, ..., orig_argN, SEGMENT_GRADS, SEGMENT_IDS
  // orig_argXs represent original op's inputs and will be passed to the reducer
  // directly
  static constexpr int kNumInputs =
      ReducerGradient::originalInputs().size() + 2;
  enum _InputTags {
    SEGMENT_GRADS = ReducerGradient::originalInputs().size(),
    SEGMENT_IDS
  };
};

// base implementation of sorted/unsorted sparse/non-sparse gradient computation
template <
    typename ForwardOp,
    typename ReducerDef,
    typename ReducerGradient,
    bool Sorted,
    bool SparseFused>
struct SegmentOpGetGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(
        !ReducerGradient::requiresDataInput(Def()),
        "grads on aux inputs are not yet implemented for Segment operators.");
    vector<string> grad_ins;
    for (const int i : ReducerGradient::originalInputs()) {
      grad_ins.push_back(I(i));
    }
    grad_ins.push_back(GO(0));
    grad_ins.push_back(I(ForwardOp::SEGMENT_IDS));
    vector<OperatorDef> r{CreateOperatorDef(
        string(Sorted ? "SortedSegment" : "UnsortedSegment") +
            ReducerDef::name + "Gradient",
        "",
        grad_ins,
        // no gradient on segment_ids or auxiliary inputs for now
        vector<string>{SparseFused ? GI_V(0) : GI(0)})};
    if (SparseFused) {
      SetSparse(0, I(ForwardOp::INDICES), GI_V(0));
    }
    return r;
  }
};

template <typename T, typename SIndex, typename Context, typename ReducerDef>
struct AbstractSortedSegmentDef {
  using OpDef = ReducerDef;
  static constexpr const char* basename = "SortedSegment";
  static constexpr const char* doc = R"DOC(
Applies '{op}' to each segment of input tensor. Segments need to be sorted and
contiguous. See also UnsortedSegment{op} that doesn't have this requirement.

SEGMENT_IDS is a vector that maps each of the first dimension slices of the
DATA to a particular group (segment). Values belonging to the same segment are
aggregated together.

The first dimension of the output is equal to the number of input segments,
i.e. `SEGMENT_IDS[-1]+1`. Other dimensions are inherited from the input tensor.

{op_doc}
  )DOC";
  static void PopulateSchema(OpSchema& schema) {
    schema.Input(0, "DATA", "Input tensor, slices of which are aggregated.");
    schema.Input(
        Reducer::kInputCount,
        "SEGMENT_IDS",
        "Vector with the same length as the first dimension of DATA "
        "and values in the range 0..K-1 and in increasing order that "
        "maps each slice of DATA to one of the segments");
    schema.Output(
        0,
        "OUTPUT",
        "Aggregated output tensor. Has the first dimension of K "
        "(the number of segments).");
    ReducerDef::PopulateSchema(schema);
  }
  using Reducer = typename ReducerDef::template Reducer<T, Context>;
  using ReducerGradient =
      typename ReducerDef::template ReducerGradient<T, Context>;
  using ForwardOp = AbstractSortedSegmentOp<T, SIndex, Context, Reducer, false>;
  using BackwardOp =
      AbstractSortedSegmentGradientOp<T, SIndex, Context, ReducerGradient>;
  using GetGradient = SegmentOpGetGradient<
      ForwardOp,
      ReducerDef,
      ReducerGradient,
      true /*Sorted*/,
      false /*SparseFused*/>;
};

template <typename T, typename SIndex, typename Context, typename ReducerDef>
struct AbstractSparseSortedSegmentDef {
  using OpDef = ReducerDef;
  static constexpr const char* basename = "SparseSortedSegment";
  static constexpr const char* doc = R"DOC(
Pulls in slices of the input tensor, groups them into segments and applies
'{op}' to each segment. Segments need to be sorted and contiguous. See also
SparseUnsortedSegment{op} that doesn't have this requirement.

This op is basically Gather and SortedSegment{op} fused together.

INDICES should contain integers in range 0..N-1 where N is the first dimension
of DATA. INDICES represent which slices of DATA need to be pulled in.

SEGMENT_IDS is a vector that maps each referenced slice of the DATA to a
particular group (segment). Values belonging to the same segment are aggregated
together. SEGMENT_IDS should have the same dimension as INDICES.

The first dimension of the output is equal to the number of input segments,
i.e. `SEGMENT_IDS[-1]+1`. Other dimensions are inherited from the input tensor.

{op_doc}
  )DOC";
  static void PopulateSchema(OpSchema& schema) {
    schema.Input(0, "DATA", "Input tensor, slices of which are aggregated.");
    schema.Input(
        Reducer::kInputCount,
        "INDICES",
        "Integer vector containing indices of the first dimension of DATA for "
        "the slices that are being aggregated");
    schema.Input(
        Reducer::kInputCount + 1,
        "SEGMENT_IDS",
        "Vector with the same length as INDICES and values in the range "
        "0..K-1 and in increasing order that maps each slice of DATA referenced"
        " by INDICES to one of the segments");
    schema.Output(
        0,
        "OUTPUT",
        "Aggregated output tensor. Has the first dimension of K "
        "(the number of segments).");
    ReducerDef::PopulateSchema(schema);
  }
  using Reducer = typename ReducerDef::template Reducer<T, Context>;
  using ReducerGradient =
      typename ReducerDef::template ReducerGradient<T, Context>;
  using ForwardOp = AbstractSortedSegmentOp<T, SIndex, Context, Reducer>;
  // TODO(dzhulgakov): we're registering the same class twice here,
  // consider avoiding op duplication here
  using BackwardOp =
      AbstractSortedSegmentGradientOp<T, SIndex, Context, ReducerGradient>;
  using GetGradient = SegmentOpGetGradient<
      ForwardOp,
      ReducerDef,
      ReducerGradient,
      true /*Sorted*/,
      true /*SparseFused*/>;
};

/**
 * @brief Unsorted segment reduction op with optional fused embedding lookup
 *
 * Base implementation for UnsortedSegmentXXX and UnsparseSortedSegmentXXX
 * depending on SparseFused static argument.
 *
 * Unlike the sorted version it allows to have "gaps" in segment ids.
 *
 * Inputs:
 *   0: DATA - input embedding to do lookups in
 *   1..P: AUX_ARG_<I> - optional additional arguments to be passed to the
 *                       reducer, should have the same first dimension as
 *                       SEGMENT_IDS (e.g. scalars in WeightedSum)
 *   # if SparseFused == true:
 *   P+1: INDICES - 1-D vector with indices to look up in DATA. Should have the
 *                  same dimension as SEGMENT_IDS
 *   # P+1 if SparseFused == false:
 *   P+1 or P+2: SEGMENT_IDS - unsorted segment ids 1-D vector
 *
 * Args:
 *   num_segments - allows to override the dimension of the output. If not set
 *                  it would be inferred from segment_ids tensor.
 *
 *
 * Output:
 *   Tensor with first dimension of K, where K is the max segment id + 1. Rest
 *   of dimensions are decided by reducer but usually are the same size as extra
 *   dimensions of DATA
 */
template <
    typename T,
    typename SIndex,
    class Context,
    class Reducer,
    bool SparseFused = true,
    class InputAccessor = BaseInputAccessor<T>>
class AbstractUnsortedSegmentOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit AbstractUnsortedSegmentOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "num_segments", num_segments_, -1) {}

  bool RunOnDevice() override {
    if (SparseFused) {
      return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
          this, Input(INDICES));
    } else {
      // type doesn't matter
      return DoRunWithType<int64_t>();
    }
  }

  template <typename IndexType>
  bool DoRunWithType() {
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    int64_t in_block_size = Input(0).size_from_dim(1);
    return DispatchHelper<typename Reducer::FixedDispatch, IndexType>::call(
        this, in_block_size);
  }

  template <typename IndexType, int FixedSize>
  bool DoRunWithValue() {
    auto& data = Input(0);
    auto& segment_ids = Input(SEGMENT_IDS);

    CAFFE_ENFORCE_EQ(1, segment_ids.dim(), "SEGMENT_IDS must be a vector");
    int64_t N = segment_ids.size(0);
    const int64_t M = data.size(0);

    const IndexType* idxs;
    if (SparseFused) { // static if
      auto& indices = Input(INDICES);
      CAFFE_ENFORCE_EQ(1, indices.dim(), "INDICES must be a vector");
      CAFFE_ENFORCE_EQ(
          N,
          indices.size(0),
          "SEGMENT_IDS must have the same length as INDICES");
      idxs = indices.template data<IndexType>();
    } else {
      CAFFE_ENFORCE_EQ(
          N, M, "DATA must have the same first dimension as SEGMENT_IDS");
    }

    // It would probably look nicer with varargs templates but it's too much
    // metaprogramming
    typename Reducer::Meta ctx;
    ctx.observeInput(0, data, 1);
    for (int i = 1; i < Reducer::kInputCount; ++i) {
      auto& aux_in = Input(i);
      CAFFE_ENFORCE_EQ(
          N,
          aux_in.size(0),
          "Input ",
          i,
          " must have the same first dim as SEGMENT_IDS");
      ctx.observeInput(i, aux_in, 1);
    }

    const SIndex* s_ids = segment_ids.template data<SIndex>();
    OPERATOR_NEEDS_FEATURE(
        inputAccessor_.observeInput(data),
        "Unsupported input type: ",
        data.dtype().name(),
        ".");

    // determine the number of segments
    SIndex K;
    if (num_segments_ != -1) {
      K = num_segments_;
    } else {
      K = 0;
      for (int64_t i = 0; i < N; ++i) {
        K = std::max(K, s_ids[i] + 1);
      }
    }

    vector<int64_t> shape;
    shape.push_back(K);
    ctx.appendOutputShape(&shape);
    auto* output = Output(0, shape, at::dtype<T>());

    int64_t in_block_size = data.size_from_dim(1);
    int64_t out_block_size = output->size_from_dim(1);
    T* out = output->template mutable_data<T>();

    reducers_.clear();
    reducers_.reserve(K);
    for (int64_t i = 0; i < K; ++i) {
      reducers_.emplace_back(ctx, out + out_block_size * i, &context_);
    }

    for (int64_t i = 0; i < N; ++i) {
      auto s_id = s_ids[i];
      CAFFE_ENFORCE(
          0 <= s_id && s_id < K,
          "Segment id out of range: ",
          s_id,
          ", range 0 to ",
          K);
      IndexType idx;
      if (SparseFused) { // static if
        CAFFE_ENFORCE(
            0 <= idxs[i] && idxs[i] < M,
            "Index out of bounds: ",
            idxs[i],
            ", range 0 to ",
            M);
        idx = idxs[i];
      } else {
        idx = i;
      }
      reducers_[s_id].template process<FixedSize>(
          ctx, inputAccessor_.getBlockPtr(in_block_size, idx), i, &context_);
    }

    for (int64_t i = 0; i < K; ++i) {
      reducers_[i].template finish<FixedSize>(ctx, &context_);
    }
    // call reducers destructors (if there is any)
    reducers_.clear();
    return true;
  }

  enum {
    INDICES = Reducer::kInputCount,
    SEGMENT_IDS = Reducer::kInputCount + (SparseFused ? 1 : 0)
  };
  static constexpr int kSelfInputs = SparseFused ? 2 : 1;
  static constexpr int kNumInputs = Reducer::kInputCount + kSelfInputs;

 private:
  int64_t num_segments_;
  // member field to reuse memory
  vector<Reducer> reducers_;
  InputAccessor inputAccessor_;
};

// Gradient actually doesn't depend on whether sparse lookup is fused or not
template <typename T, typename SIndex, class Context, class ReducerGradient>
class AbstractUnsortedSegmentGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AbstractUnsortedSegmentGradientOp);

  bool RunOnDevice() override {
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    int64_t grad_block_size = Input(SEGMENT_GRADS).size_from_dim(1);
    return DispatchHelper<typename ReducerGradient::FixedDispatch>::call(
        this, grad_block_size);
  }

  template <int FixedSize>
  bool DoRunWithValue() {
    auto& segment_grads = Input(SEGMENT_GRADS);
    auto& segment_ids = Input(SEGMENT_IDS);

    CAFFE_ENFORCE_EQ(1, segment_ids.dim(), "SEGMENT_IDS must be a vector");
    int64_t N = segment_ids.size(0);

    typename ReducerGradient::Meta ctx(segment_grads, 1);
    for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
      auto& aux_in = Input(i);
      CAFFE_ENFORCE_EQ(
          N,
          aux_in.size(0),
          "Input ",
          i,
          " must have the same first dim as SEGMENT_IDS");
      ctx.observeOriginalInput(
          ReducerGradient::originalInputs()[i], aux_in, nullptr /*no grad*/, 1);
    }

    const SIndex* s_ids = segment_ids.template data<SIndex>();
    const T* s_grads = segment_grads.template data<T>();

    vector<int64_t> shape;
    shape.push_back(N);
    ctx.appendGradShape(&shape);
    auto* data_grads = Output(0, shape, at::dtype<T>());

    int64_t d_block_size = data_grads->size_from_dim(1);
    const SIndex K = segment_grads.size(0);
    int64_t s_block_size = segment_grads.size_from_dim(1);
    T* out = data_grads->template mutable_data<T>();

    if (ReducerGradient::computeLength()) {
      segment_length_.resize(K, 0);
      for (int i = 0; i < N; ++i) {
        auto s_id = s_ids[i];
        CAFFE_ENFORCE(
            0 <= s_id && s_id < K,
            "Segment id out of range: ",
            s_id,
            ", range 0 to ",
            K);
        segment_length_[s_ids[i]]++;
      }
    }

    reducers_.clear();
    reducers_.reserve(K);
    for (SIndex i = 0; i < K; ++i) {
      reducers_.emplace_back(ctx, s_grads + s_block_size * i, &context_);
    }

    for (int64_t i = 0; i < N; ++i) {
      auto s_id = s_ids[i];
      if (ReducerGradient::computeLength()) {
        reducers_[s_id].template fillGrad<FixedSize>(
            ctx, out + d_block_size * i, i, &context_, segment_length_[s_id]);
      } else {
        reducers_[s_id].template fillGrad<FixedSize>(
            ctx, out + d_block_size * i, i, &context_, 0);
      }
    }
    // call reducers destructors (if there is any)
    reducers_.clear();
    return true;
  }

  // Input layout:
  //   orig_arg1, orig_arg2, ..., orig_argN, SEGMENT_GRADS, SEGMENT_IDS
  // orig_argXs represent original op's inputs and will be passed to the reducer
  // directly
  static constexpr int kNumInputs =
      ReducerGradient::originalInputs().size() + 2;
  enum _InputTags {
    SEGMENT_GRADS = ReducerGradient::originalInputs().size(),
    SEGMENT_IDS
  };

 private:
  // member field to reuse memory
  vector<ReducerGradient> reducers_;
  vector<int> segment_length_;
};

template <typename T, typename SIndex, typename Context, typename ReducerDef>
struct AbstractUnsortedSegmentDef {
  using OpDef = ReducerDef;
  static constexpr const char* basename = "UnsortedSegment";
  static constexpr const char* doc = R"DOC(
Applies '{op}' to each segment of input tensor. Segments ids can appear in
arbitrary order (unlike in SortedSegment{op}).

SEGMENT_IDS is a vector that maps each of the first dimension slices of the
DATA to a particular group (segment). Values belonging to the same segment are
aggregated together.

If `num_segments` argument is passed it would be used as a first dimension for
the output. Otherwise, it'd be dynamically calculated from as the max value of
SEGMENT_IDS plus one. Other output dimensions are inherited from the input
tensor.

{op_doc}
  )DOC";
  static void PopulateSchema(OpSchema& schema) {
    schema.Arg(
        "num_segments",
        "Optional int argument specifying the number of output segments and "
        "thus the first dimension of the output");
    schema.Input(0, "DATA", "Input tensor, slices of which are aggregated.");
    schema.Input(
        Reducer::kInputCount,
        "SEGMENT_IDS",
        "Integer vector with the same length as the first dimension of DATA "
        "that maps each slice of DATA to one of the segments");
    schema.Output(
        0,
        "OUTPUT",
        "Aggregated output tensor. Has the first dimension of equal to the "
        "number of segments.");
    ReducerDef::PopulateSchema(schema);
  }
  using Reducer = typename ReducerDef::template Reducer<T, Context>;
  using ReducerGradient =
      typename ReducerDef::template ReducerGradient<T, Context>;
  using ForwardOp = AbstractUnsortedSegmentOp<
      T,
      SIndex,
      Context,
      typename ReducerDef::template Reducer<T, Context>,
      false>;
  using BackwardOp =
      AbstractUnsortedSegmentGradientOp<T, SIndex, Context, ReducerGradient>;
  using GetGradient = SegmentOpGetGradient<
      ForwardOp,
      ReducerDef,
      ReducerGradient,
      false /*Sorted*/,
      false /*SparseFused*/>;
};

template <typename T, typename SIndex, typename Context, typename ReducerDef>
struct AbstractSparseUnsortedSegmentDef {
  using OpDef = ReducerDef;
  static constexpr const char* basename = "SparseUnsortedSegment";
  static constexpr const char* doc = R"DOC(
Pulls in slices of the input tensor, groups them into segments and applies
'{op}' to each segment. Segments ids can appear in arbitrary order (unlike in
SparseSortedSegment{op}).

This op is basically Gather and UnsortedSegment{op} fused together.

INDICES should contain integers in range 0..N-1 where N is the first dimension
of DATA. INDICES represent which slices of DATA need to be pulled in.

SEGMENT_IDS is a vector that maps each referenced slice of the DATA to a
particular group (segment). Values belonging to the same segment are aggregated
together. SEGMENT_IDS should have the same dimension as INDICES.

If `num_segments` argument is passed it would be used as a first dimension for
the output. Otherwise, it'd be dynamically calculated from as the max value of
SEGMENT_IDS plus one. Other output dimensions are inherited from the input
tensor.

{op_doc}
  )DOC";
  static void PopulateSchema(OpSchema& schema) {
    schema.Input(0, "DATA", "Input tensor, slices of which are aggregated.");
    schema.Input(
        Reducer::kInputCount,
        "INDICES",
        "Integer vector containing indices of the first dimension of DATA for "
        "the slices that are being aggregated");
    schema.Input(
        Reducer::kInputCount + 1,
        "SEGMENT_IDS",
        "Integer vector with the same length as INDICES that maps each slice "
        "of DATA referenced by INDICES to one of the segments");
    schema.Output(
        0,
        "OUTPUT",
        "Aggregated output tensor. Has the first dimension of equal to the "
        "number of segments.");
    ReducerDef::PopulateSchema(schema);
  }
  using Reducer = typename ReducerDef::template Reducer<T, Context>;
  using ReducerGradient =
      typename ReducerDef::template ReducerGradient<T, Context>;
  using ForwardOp = AbstractUnsortedSegmentOp<T, SIndex, Context, Reducer>;
  // TODO(dzhulgakov): we're registering the same class twice here,
  // consider avoiding op duplication here
  using BackwardOp =
      AbstractUnsortedSegmentGradientOp<T, SIndex, Context, ReducerGradient>;
  using GetGradient = SegmentOpGetGradient<
      ForwardOp,
      ReducerDef,
      ReducerGradient,
      false /*Sorted*/,
      true /*SparseFused*/>;
};

/**
 * @brief Segment reduction op with optional fused embedding lookup
 *
 * Base implementation for LengthsXXX and SparseLengthsXXX depending
 * on SparseFused static argument.
 *
 * Inputs:
 *   0: DATA - input embedding to do lookups in
 *   1..P: AUX_ARG_<I> - optional additional arguments to be passed to the
 *                       reducer, should have the same first dimension as
 *                       LENGTHS (e.g. scalars in WeightedSum)
 *   # if SparseFused == true:
 *   P+1: INDICES - 1-D vector with indices to look up in DATA. Should have the
 *                  same dimension as LENGTHS
 *   # P+1 if SparseFused == false:
 *   P+1 or P+2: LENGTHS - lengths on indecies vector
 *
 * Output:
 *   Tensor with first dimension of K, where K = len(LENGTHS). Rest
 *   of dimensions are decided by reducer but usually are the same size as extra
 *   dimensions of DATA
 */
// TODO(dzhulgakov): for now it's implemented with incremental reducers because
// of fused sparse support. But using "lengths" representation actually implies
// continuous segments and thus range reducers can be used for non-sparse
// version.

template <
    typename TData,
    typename TLengths,
    class Context,
    class Reducer,
    bool SparseFused = true,
    class InputAccessor = BaseInputAccessor<TData>>
class AbstractLengthsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AbstractLengthsOp);

  bool RunOnDevice() override {
    if (SparseFused) {
      return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
          this, Input(INDICES));
    } else {
      // type doesn't matter
      return DoRunWithType<int64_t>();
    }
  }

  template <typename IndexType>
  bool DoRunWithType() {
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    int64_t in_block_size = Input(0).size_from_dim(1);
    return DispatchHelper<typename Reducer::FixedDispatch, IndexType>::call(
        this, in_block_size);
  }

  template <typename IndexType, int FixedSize>
  bool DoRunWithValue() {
    auto& dataInput = Input(0);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    const int64_t dataSize = dataInput.size(0);
    // Either first dim the data or how much we pull in indexies from it
    int64_t dataToReduceSize;
    const int64_t outputSize = lengthsInput.size(0);

    const IndexType* indices;
    if (SparseFused) { // static if
      auto& indicesInput = Input(INDICES);
      CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
      indices = indicesInput.template data<IndexType>();
      dataToReduceSize = indicesInput.size(0);
    } else {
      dataToReduceSize = dataSize;
    }

    typename Reducer::Meta ctx;
    ctx.observeInput(0, dataInput, 1);
    for (int i = 1; i < Reducer::kInputCount; ++i) {
      auto& aux_in = Input(i);
      CAFFE_ENFORCE(
          dataToReduceSize == aux_in.size(0),
          "Input ",
          i,
          " must have the same first dim as SEGMENT_IDS");
      ctx.observeInput(i, aux_in, 1);
    }

    const TLengths* lengths = lengthsInput.template data<TLengths>();

    OPERATOR_NEEDS_FEATURE(
        inputAccessor_.observeInput(dataInput),
        "Unsupported input type: ",
        dataInput.dtype().name(),
        ".");

    vector<int64_t> shape{outputSize};
    ctx.appendOutputShape(&shape);
    auto* output = Output(0, shape, at::dtype<TData>());

    int64_t in_block_size = dataInput.size_from_dim(1);
    int64_t out_block_size = output->size_from_dim(1);
    TData* out = output->template mutable_data<TData>();

    int64_t dataIndex = 0;
    for (int64_t rangeIndex = 0; rangeIndex < outputSize; ++rangeIndex) {
      Reducer reducer(ctx, out + out_block_size * rangeIndex, &context_);
      for (int64_t start = dataIndex; dataIndex < start + lengths[rangeIndex];
           ++dataIndex) {
        IndexType idx;
        if (SparseFused) { // static if
          idx = indices[dataIndex];
          CAFFE_ENFORCE(
              0 <= idx && idx < dataSize,
              "The ",
              dataIndex,
              "th index from the input indices is out of bounds: ",
              idx,
              " vs. valid range 0 to ",
              dataSize);
        } else {
          idx = dataIndex;
          CAFFE_ENFORCE(
              0 <= idx && idx < dataSize,
              "When calculating the ",
              rangeIndex,
              "th output with length=",
              lengths[rangeIndex],
              ", the index is out of bounds: ",
              idx,
              " vs. valid range 0 to ",
              dataSize);
        }

        const TData* input = inputAccessor_.getBlockPtr(in_block_size, idx);
        reducer.template process<FixedSize>(ctx, input, dataIndex, &context_);
      }
      reducer.template finish<FixedSize>(ctx, &context_);
    }
    CAFFE_ENFORCE(
        dataIndex == dataToReduceSize, dataIndex, " != ", dataToReduceSize);

    return true;
  }

  enum {
    INDICES = Reducer::kInputCount,
    LENGTHS = Reducer::kInputCount + (SparseFused ? 1 : 0)
  };
  static constexpr int kSelfInputs = SparseFused ? 2 : 1;
  static constexpr int kNumInputs = Reducer::kInputCount + kSelfInputs;

 private:
  InputAccessor inputAccessor_;
};

/*
 * Some notice:
 * 1. Gradient actually doesn't depend on whether sparse lookup is fused or not
 * 2. INDICES are not used in CPU version, but they are needed in async CUDA
 *    version. So we register 3 input version for CPU as gradient op for
 *    GPU/CPU convert. We then register 2 input version for CPU for backward
 *    compatibility with older nets.
 */
template <
    typename T,
    typename TLengths,
    class Context,
    class ReducerGradient,
    bool GradientNeedIndices = false>
class AbstractLengthsGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AbstractLengthsGradientOp);

  bool RunOnDevice() override {
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    int64_t gradBlockSize = Input(SEGMENT_GRADS).size_from_dim(1);
    return DispatchHelper<typename ReducerGradient::FixedDispatch>::call(
        this, gradBlockSize);
  }

  template <int FixedSize>
  bool DoRunWithValue() {
    auto& segmentGradsInput = Input(SEGMENT_GRADS);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE(lengthsInput.dim() == 1, "LENGTHS must be a vector");
    int64_t reducedDataSize = 0;
    int64_t numSegments = lengthsInput.size(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(numSegments == segmentGradsInput.size(0));
    const TLengths* lengths = lengthsInput.template data<TLengths>();
    for (int64_t i = 0; i < numSegments; ++i) {
      reducedDataSize += lengths[i];
    }

    typename ReducerGradient::Meta ctx(segmentGradsInput, 1);
    for (auto i = 0U; i < ReducerGradient::originalInputs().size(); ++i) {
      auto& aux_in = Input(i);
      CAFFE_ENFORCE_EQ(
          reducedDataSize,
          aux_in.size(0),
          "Input ",
          i,
          " must have the same first dim as SEGMENT_IDS");
      ctx.observeOriginalInput(
          ReducerGradient::originalInputs()[i], aux_in, nullptr /*no grad*/, 1);
    }

    const T* segmentGrads = segmentGradsInput.template data<T>();

    vector<int64_t> shape;
    shape.push_back(reducedDataSize);
    ctx.appendGradShape(&shape);
    auto* dataGradsOutput = Output(0, shape, at::dtype<T>());

    int64_t dataGradsBlockSize = dataGradsOutput->size_from_dim(1);
    int64_t segmentBlockSize = segmentGradsInput.size_from_dim(1);
    T* dataGrads = dataGradsOutput->template mutable_data<T>();

    int64_t dataIndex = 0;
    for (int64_t rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
      ReducerGradient reducer(
          ctx, segmentGrads + segmentBlockSize * rangeIndex, &context_);
      for (int64_t start = dataIndex; dataIndex < start + lengths[rangeIndex];
           ++dataIndex) {
        reducer.template fillGrad<FixedSize>(
            ctx,
            dataGrads + dataGradsBlockSize * dataIndex,
            dataIndex,
            &context_,
            lengths[rangeIndex]);
      }
    }
    CAFFE_ENFORCE(
        dataIndex == reducedDataSize, dataIndex, " != ", reducedDataSize);
    return true;
  }

  // Input layout:
  //   orig_arg1, orig_arg2, ..., orig_argN, SEGMENT_GRADS, LENGTHS, INDICES
  // orig_argXs represent original op's inputs and will be passed to the reducer
  // directly
  static constexpr int kNumInputs = ReducerGradient::originalInputs().size() +
      2 + (GradientNeedIndices ? 1 : 0);
  enum _InputTags {
    SEGMENT_GRADS = ReducerGradient::originalInputs().size(),
    LENGTHS,
    INDICES
  };
};

// Version of gradient that requires the main input and thus needs to receive
// length, indices and other stuff
template <
    typename Tembedding,
    typename T,
    typename TLengths,
    class Context,
    class ReducerGradient,
    bool SparseFused = true,
    bool GradientNeedIndices = false>
class AbstractLengthsWithMainInputGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AbstractLengthsWithMainInputGradientOp);

  bool RunOnDevice() override {
    if (SparseFused) {
      return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
          this, Input(INDICES));
    } else {
      // type doesn't matter
      return DoRunWithType<int64_t>();
    }
  }

  template <typename IndexType>
  bool DoRunWithType() {
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    int64_t in_block_size = Input(SEGMENT_GRADS).size_from_dim(1);
    return DispatchHelper<typename ReducerGradient::FixedDispatch, IndexType>::
        call(this, in_block_size);
  }

  template <typename IndexType, int FixedSize>
  bool DoRunWithValue() {
    auto& dataInput = Input(DATA_INPUT);
    auto& segmentGradsInput = Input(SEGMENT_GRADS);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE(lengthsInput.dim() == 1, "LENGTHS must be a vector");
    int64_t numSegments = lengthsInput.size(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(numSegments == segmentGradsInput.size(0));
    const TLengths* lengths = lengthsInput.template data<TLengths>();

    typename ReducerGradient::Meta ctx(segmentGradsInput, 1);
    for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
      int aux_num = ReducerGradient::originalInputs()[i];
      auto& aux_in = Input(i);
      auto* aux_grad = aux_num < OutputSize() ? Output(aux_num) : nullptr;
      ctx.observeOriginalInput(aux_num, aux_in, aux_grad, 1);
    }

    // Either first dim the data or how much we pull in indexies from it
    int64_t dataToReduceSize;
    const IndexType* indices = nullptr;
    if (SparseFused) { // static if
      auto& indicesInput = Input(INDICES);
      indices = indicesInput.template data<IndexType>();
      dataToReduceSize = indicesInput.size(0);
    } else {
      dataToReduceSize = dataInput.size(0);
    }

    const T* segmentGrads = segmentGradsInput.template data<T>();

    vector<int64_t> shape;
    shape.push_back(dataToReduceSize);
    ctx.appendGradShape(&shape);
    auto* dataGradsOutput = Output(0, shape, at::dtype<T>());

    int64_t dataGradsBlockSize = dataGradsOutput->size_from_dim(1);
    int64_t segmentBlockSize = segmentGradsInput.size_from_dim(1);
    T* dataGrads = dataGradsOutput->template mutable_data<T>();

    const Tembedding* data = dataInput.template data<Tembedding>();
    int64_t dataIndex = 0;
    for (int64_t rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
      ReducerGradient reducer(
          ctx, segmentGrads + segmentBlockSize * rangeIndex, &context_);
      for (int64_t start = dataIndex; dataIndex < start + lengths[rangeIndex];
           ++dataIndex) {
        IndexType data_pos;
        // No range checking, should've been verified in forward pass
        if (SparseFused) { // static if
          data_pos = indices[dataIndex];
        } else {
          data_pos = dataIndex;
        }
        reducer.template fillGradWithMainInput<FixedSize>(
            ctx,
            data + dataGradsBlockSize * data_pos,
            dataGrads + dataGradsBlockSize * dataIndex,
            dataIndex,
            &context_,
            lengths[rangeIndex]);
      }
    }
    return true;
  }

  // Input layout:
  //   orig_arg1, orig_arg2, ..., orig_argN, SEGMENT_GRADS, LENGTHS,
  //      DATA_INPUT, [INDICES]
  // orig_argXs represent original op's inputs and will be passed to the reducer
  // directly
  static constexpr int kNumInputs = ReducerGradient::originalInputs().size() +
      3 + (SparseFused ? 1 : 0) + (GradientNeedIndices ? 1 : 0);
  enum _InputTags {
    SEGMENT_GRADS = ReducerGradient::originalInputs().size(),
    LENGTHS,
    DATA_INPUT,
    INDICES,
  };
};

// Version of gradient that requires the main input as well as the output of the
// forward op.
template <typename T, typename TLengths, class Context, class ReducerGradient>
class AbstractLengthsWithMainInputAndForwardOutputGradientOp
    : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AbstractLengthsWithMainInputAndForwardOutputGradientOp);

  bool RunOnDevice() override {
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class.
    int64_t in_block_size = Input(SEGMENT_GRADS).size_from_dim(1);
    return DispatchHelper<typename ReducerGradient::FixedDispatch>::call(
        this, in_block_size);
  }

  template <int FixedSize>
  bool DoRunWithValue() {
    auto& dataInput = Input(DATA_INPUT);
    auto& segmentGradsInput = Input(SEGMENT_GRADS);
    auto& lengthsInput = Input(LENGTHS);
    auto& forwardOutputInput = Input(FORWARD_OUTPUT);

    CAFFE_ENFORCE(lengthsInput.dim() == 1, "LENGTHS must be a vector");
    int64_t numSegments = lengthsInput.size(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(numSegments == segmentGradsInput.size(0));
    const TLengths* lengths = lengthsInput.template data<TLengths>();

    typename ReducerGradient::Meta ctx(segmentGradsInput, 1);
    for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
      int aux_num = ReducerGradient::originalInputs()[i];
      auto& aux_in = Input(i);
      auto* aux_grad = aux_num < OutputSize() ? Output(aux_num) : nullptr;
      ctx.observeOriginalInput(aux_num, aux_in, aux_grad, 1);
    }

    CAFFE_ENFORCE(forwardOutputInput.dim() > 0);
    CAFFE_ENFORCE(numSegments == forwardOutputInput.size(0));
    const T* forwardOutput = forwardOutputInput.template data<T>();

    int64_t dataToReduceSize = dataInput.size(0);

    const T* segmentGrads = segmentGradsInput.template data<T>();

    vector<int64_t> shape;
    shape.push_back(dataToReduceSize);
    ctx.appendGradShape(&shape);
    auto* dataGradsOutput = Output(0, shape, at::dtype<T>());

    int64_t dataGradsBlockSize = dataGradsOutput->size_from_dim(1);
    int64_t segmentBlockSize = segmentGradsInput.size_from_dim(1);
    T* dataGrads = dataGradsOutput->template mutable_data<T>();

    const T* data = dataInput.template data<T>();

    int64_t dataIndex = 0;
    for (int64_t rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
      ReducerGradient reducer(
          ctx, segmentGrads + segmentBlockSize * rangeIndex, &context_);
      for (int64_t start = dataIndex; dataIndex < start + lengths[rangeIndex];
           ++dataIndex) {
        // No range checking, should've been verified in forward pass
        reducer.template fillGradWithMainInputAndForwardOutput<FixedSize>(
            ctx,
            data + dataGradsBlockSize * dataIndex,
            dataGrads + dataGradsBlockSize * dataIndex,
            forwardOutput + segmentBlockSize * rangeIndex,
            dataIndex,
            &context_,
            lengths[rangeIndex]);
      }
    }
    return true;
  }

  // Input layout:
  //   orig_arg1, orig_arg2, ..., orig_argN, FORWARD_OUTPUT, SEGMENT_GRADS,
  //      LENGTHS, DATA_INPUT
  // orig_argXs represent original op's inputs and will be passed to the reducer
  // directly
  static constexpr int kNumInputs =
      ReducerGradient::originalInputs().size() + 4;
  enum _InputTags {
    FORWARD_OUTPUT = ReducerGradient::originalInputs().size(),
    SEGMENT_GRADS,
    LENGTHS,
    DATA_INPUT,
  };
};

// base implementation of sparse/non-sparse gradient computation
template <
    typename ForwardOp,
    typename ReducerDef,
    typename ReducerGradient,
    bool SparseFused,
    bool GradientNeedIndices = false>
struct LengthsOpGetGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> grad_ins;
    string suffix = "Gradient";
    for (const int i : ReducerGradient::originalInputs()) {
      grad_ins.push_back(I(i));
    }
    if (ReducerGradient::requiresForwardOutput()) {
      grad_ins.push_back(O(0));
      CAFFE_ENFORCE(
          !SparseFused,
          "Forward pass output not yet supported as input for backward pass "
          "for SparseLengthsXXX operators");
      suffix = "AndForwardOutput" + suffix;
    }
    grad_ins.push_back(GO(0));
    grad_ins.push_back(I(ForwardOp::LENGTHS));
    bool indices_pushed = false;
    if (ReducerGradient::requiresDataInput(Def())) {
      grad_ins.push_back(I(0));
      if (SparseFused) {
        grad_ins.push_back(I(ForwardOp::INDICES));
        indices_pushed = true;
      }
      suffix = "WithMainInput" + suffix;
    }
    if (GradientNeedIndices && !indices_pushed) {
      if (SparseFused) {
        grad_ins.push_back(I(ForwardOp::INDICES));
      } else {
        // Hacky: using Input as Indices, remove this after we have specialized
        // cuda LengthsIndicesInGradientSumGradient
        grad_ins.push_back(I(0));
      }
    }
    vector<string> grad_outs;
    grad_outs.push_back({SparseFused ? GI_V(0) : GI(0)});
    int aux_grads = ReducerGradient::numAuxInputsWithGrads(Def());
    for (int i = 1; i <= aux_grads; ++i) {
      grad_outs.push_back(GI(i));
    }
    vector<OperatorDef> r{CreateOperatorDef(
        string(SparseFused ? "SparseLengths" : "Lengths") +
            string(GradientNeedIndices ? "IndicesInGradient" : "") +
            ReducerDef::name + suffix,
        "",
        grad_ins,
        grad_outs)};
    if (SparseFused) {
      SetSparse(0, I(ForwardOp::INDICES), GI_V(0));
    }
    return r;
  }
};

template <
    typename T,
    typename SIndex,
    typename Context,
    typename ReducerDef,
    bool GradientNeedIndices = false>
struct AbstractLengthsDef {
  using OpDef = ReducerDef;
  static constexpr const char* basename = "Lengths";
  static constexpr const char* doc = R"DOC(
Applies '{op}' to each segment of the input tensor. Segments are defined
by their *LENGTHS*. *LENGTHS* is a vector that maps each of the slices of
*DATA* to a particular segment. Values belonging to the same segment are
aggregated together and considered for the '{op}' operation.

For example *LENGTHS = [2, 1]* stands for segments *DATA[0..1]* and *DATA[2]*

The sum of elements in *LENGTHS* must equal the number of elements in the first
dimension of *DATA*. The length of *OUTPUT* is equal to the number of input
segments, i.e. len(*LENGTHS*).

{op_doc}

{extra}
  )DOC";
  static void PopulateSchema(OpSchema& schema) {
    schema.Input(0, "DATA", "Input tensor, slices of which are aggregated.");
    schema.Input(
        Reducer::kInputCount,
        "LENGTHS",
        "Vector with the same sum of elements as the first dimension of DATA");
    schema.Output(
        0,
        "OUTPUT",
        "Aggregated output tensor. Has the first dimension of len(LENGTHS) ");
    schema.TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          vector<TensorShape> out(0);
          TensorShape output;
          for (int d : in[Reducer::kInputCount].dims()) {
            output.add_dims(d);
          }
          for (int j = 1; j < in[0].dims_size(); j++) {
            output.add_dims(in[0].dims(j));
          }
          output.set_data_type(in[0].data_type());
          out.push_back(output);
          return out;
        });
    ReducerDef::PopulateSchema(schema);
  }
  using Reducer = typename ReducerDef::template Reducer<T, Context>;
  using ReducerGradient =
      typename ReducerDef::template ReducerGradient<T, Context>;
  using ForwardOp = AbstractLengthsOp<T, SIndex, Context, Reducer, false>;
  using BackwardOp =
      AbstractLengthsGradientOp<T, SIndex, Context, ReducerGradient>;
  using WithMainInputBackwardOp = AbstractLengthsWithMainInputGradientOp<
      T,
      T,
      SIndex,
      Context,
      ReducerGradient,
      false>;
  using WithMainInputAndForwardOutputBackwardOp =
      AbstractLengthsWithMainInputAndForwardOutputGradientOp<
          T,
          SIndex,
          Context,
          ReducerGradient>;
  using GetGradient = LengthsOpGetGradient<
      ForwardOp,
      ReducerDef,
      ReducerGradient,
      false /*SparseFused*/,
      GradientNeedIndices>;
};

OpSchema::Cost CostInferenceForSparseLengths(
    const OperatorDef& def,
    const vector<TensorShape>& inputs,
    bool use_weight);

template <
    typename T,
    typename SIndex,
    typename Context,
    typename ReducerDef,
    bool GradientNeedIndices = false>
struct AbstractSparseLengthsDef {
  using OpDef = ReducerDef;
  static constexpr const char* basename = "SparseLengths";
  static constexpr const char* doc = R"DOC(
Pulls in slices of the input tensor, groups them into segments and applies
'{op}' to each segment. Segments are defined by their LENGTHS.

This op is basically Gather and Lengths{op} fused together.

INDICES should contain integers in range 0..N-1 where N is the first dimension
of DATA. INDICES represent which slices of DATA need to be pulled in.

LENGTHS is a vector that defines slice sizes by first dimension of DATA. Values
belonging to the same segment are aggregated together. sum(LENGTHS) has
to match INDICES size.

The first dimension of the output is equal to the number of input segment,
i.e. `len(LENGTHS)`. Other dimensions are inherited from the input tensor.

{op_doc}
  )DOC";
  static void PopulateSchema(OpSchema& schema) {
    schema.Input(0, "DATA", "Input tensor, slices of which are aggregated.");
    schema.Input(
        Reducer::kInputCount,
        "INDICES",
        "Integer vector containing indices of the first dimension of DATA for "
        "the slices that are being aggregated");
    schema.Input(
        Reducer::kInputCount + 1,
        "LENGTHS",
        "Non negative vector with sum of elements equal to INDICES length");
    schema.Output(
        0,
        "OUTPUT",
        "Aggregated output tensor. Has the first dimension of K "
        "(the number of segments).");
    schema.TensorInferenceFunction(
        [](const OperatorDef&, const std::vector<TensorShape>& input_types) {
          std::vector<TensorShape> out(1);
          out[0] = input_types[0];
          out[0].set_dims(0, input_types[Reducer::kInputCount + 1].dims(0));
          return out;
        });
    ReducerDef::PopulateSchema(schema);

    schema.CostInferenceFunction(
        [](const OperatorDef& def,
           const vector<TensorShape>& inputs) -> OpSchema::Cost {
          return CostInferenceForSparseLengths(
              def, inputs, strcmp(OpDef::name, "WeightedSum") == 0);
        });
  }
  using Reducer = typename ReducerDef::template Reducer<T, Context>;
  using ReducerGradient =
      typename ReducerDef::template ReducerGradient<T, Context>;
  using ForwardOp = AbstractLengthsOp<T, SIndex, Context, Reducer>;
  // TODO(dzhulgakov): we're registering the same class twice here,
  // consider avoiding op duplication here
  // Note: registering 2 input version for now because of naming in the macro,
  // will register 3 input version alone
  /* INDICES are not used in CPU version, but they are needed in async CUDA
   *    version. So we register 3 input version for CPU as gradient op for
   *    GPU/CPU convert. We then register 2 input version for CPU for backward
   *    compatibility with older nets.
   */
  using BackwardOp = AbstractLengthsGradientOp<
      T,
      SIndex,
      Context,
      ReducerGradient,
      false /*GradientNeedIndices*/>;
  using WithMainInputBackwardOp = AbstractLengthsWithMainInputGradientOp<
      T,
      T,
      SIndex,
      Context,
      ReducerGradient>;
  // Will return 3 input version. This is aligning new CPU/GPU nets.
  using GetGradient = LengthsOpGetGradient<
      ForwardOp,
      ReducerDef,
      ReducerGradient,
      true /*SparseFused*/,
      GradientNeedIndices>;
};
} // namespace caffe2

#endif // CAFFE2_OPERATORS_SEGMENT_REDUCTION_OP_H_
