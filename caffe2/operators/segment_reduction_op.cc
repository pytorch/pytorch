#include <cstdio>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/reducer_functors.h"

namespace caffe2 {

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
template <typename T, typename SIndex, class Context, class RangeReducer>
class AbstractSortedSegmentRangeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AbstractSortedSegmentRangeOp);

  bool RunOnDevice() override {
    auto& data = Input(DATA);
    auto& segment_ids = Input(SEGMENT_IDS);
    auto* output = Output(0);

    CHECK_EQ(1, segment_ids.ndim()) << "SEGMENT_IDS must be a vector";
    auto N = segment_ids.dim(0);
    CHECK_EQ(N, data.dim(0))
        << "SEGMENT_IDS must have the same length as outer dimension of DATA";

    const SIndex* s_ids = segment_ids.template data<SIndex>();
    const T* d = data.template data<T>();

    CHECK_GT(N, 0);
    const SIndex K = s_ids[N - 1] + 1;
    auto shape = data.dims();
    shape[0] = K;
    output->Resize(shape);

    TIndex block_size = data.size() / N;
    T* out = output->template mutable_data<T>();

    // Assume the segments are sorted and there are no gaps
    CHECK_EQ(0, s_ids[0]) << "Indices must be sorted and not have gaps";
    for (TIndex i = 0; i < N;) {
      TIndex start = i;
      for (++i; i < N && s_ids[start] == s_ids[i]; ++i)
        ;

      RangeReducer()(
          block_size,
          i - start,
          d + block_size * start,
          out + block_size * s_ids[start],
          &context_);

      // check correctness of the next segment
      if (i < N) {
        CHECK_EQ(s_ids[start] + 1, s_ids[i])
            << "Indices must be sorted and not have gaps";
      }
    }
    return true;
  }

  static constexpr int kNumInputs = 2;
  INPUT_TAGS(DATA, SEGMENT_IDS);
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
    auto* data_grads = Output(0);

    CHECK_EQ(1, segment_ids.ndim()) << "SEGMENT_IDS must be a vector";
    TIndex N = segment_ids.dim(0);

    const SIndex* s_ids = segment_ids.template data<SIndex>();
    const T* s_grads = segment_grads.template data<T>();
    const T* d_in = data_in.template data<T>();
    const T* d_out = data_out.template data<T>();

    auto shape = segment_grads.dims();
    shape[0] = N;
    data_grads->Resize(shape);

    const SIndex K = segment_grads.dim(0);
    TIndex block_size = segment_grads.size() / K;
    T* out = data_grads->template mutable_data<T>();

    // Assume the segments are sorted and there are no gaps
    CHECK_EQ(0, s_ids[0]) << "Indices must be sorted and not have gaps";
    // repeat the check from forward op
    CHECK_EQ(K - 1, s_ids[N - 1]) << "Indices must be sorted and not have gaps";
    for (TIndex i = 0; i < N;) {
      TIndex start = i;
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
        CHECK_EQ(s_ids[start] + 1, s_ids[i])
            << "Indices must be sorted and not have gaps";
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
template <typename T, class Context, class Reducer>
class AbstractReduceFrontOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  AbstractReduceFrontOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "num_reduce_dim", num_reduce_dims_, 1) {}

  bool RunOnDevice() override {
    auto& data = Input(0);
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    TIndex in_block_size = data.size_from_dim(num_reduce_dims_);
    return DispatchHelper<typename Reducer::FixedDispatch>::call(
        this, in_block_size);
  }

  template <int FixedSize>
  bool DoRunWithValue() {
    auto& data = Input(0);
    auto* output = Output(0);

    CHECK_LE(num_reduce_dims_, data.ndim());

    typename Reducer::Meta ctx;
    ctx.observeInput(0, data, num_reduce_dims_);
    for (int i = 1; i < Reducer::kInputCount; ++i) {
      auto& aux_in = Input(i);
      ctx.observeInput(i, aux_in, num_reduce_dims_);
    }

    const T* d = data.template data<T>();

    vector<TIndex> shape;
    ctx.appendOutputShape(&shape);
    output->Resize(shape);

    TIndex in_block_size = data.size_from_dim(num_reduce_dims_);
    TIndex block_num = data.size() / in_block_size;
    T* out = output->template mutable_data<T>();

    Reducer r(ctx, out, &context_);
    for (TIndex i = 0; i < block_num; ++i) {
      r.template process<FixedSize>(ctx, d + in_block_size * i, i, &context_);
    }
    return true;
  }

  static constexpr int kNumInputs = Reducer::kInputCount;

 private:
  int num_reduce_dims_;
};

template <typename T, class Context, class ReducerGradient>
class AbstractReduceFrontGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  AbstractReduceFrontGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "num_reduce_dim", num_reduce_dims_, 1) {}

  bool RunOnDevice() override {
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    TIndex grad_block_size = Input(REDUCTION_GRAD).size();
    return DispatchHelper<typename ReducerGradient::FixedDispatch>::call(
        this, grad_block_size);
  }

  template <int FixedSize>
  bool DoRunWithValue() {
    auto& reduction_grad = Input(REDUCTION_GRAD);
    auto& source_shape = OperatorBase::Input<TensorCPU>(SOURCE_SHAPE);

    auto* data_grads = Output(0);

    typename ReducerGradient::Meta ctx(reduction_grad, 0);
    for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
      auto& aux_in = Input(i);
      ctx.observeOriginalInput(
          ReducerGradient::originalInputs()[i], aux_in, num_reduce_dims_);
    }

    const T* r_grad = reduction_grad.template data<T>();

    CHECK_LE(num_reduce_dims_, source_shape.size());
    vector<TIndex> shape(
        source_shape.template data<TIndex>(),
        source_shape.template data<TIndex>() + num_reduce_dims_);
    ctx.appendGradShape(&shape);
    data_grads->Resize(shape);

    TIndex block_size = data_grads->size_from_dim(num_reduce_dims_);
    TIndex block_num = data_grads->size() / block_size;
    T* out = data_grads->template mutable_data<T>();

    ReducerGradient r(ctx, r_grad, &context_);
    for (TIndex i = 0; i < block_num; ++i) {
      r.template fillGrad<FixedSize>(ctx, out + block_size * i, i, &context_);
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
    ReducerDef::PopulateSchema(schema);
  }
  using ReducerGradient =
      typename ReducerDef::template ReducerGradient<T, Context>;
  using ForwardOp = AbstractReduceFrontOp<
      T,
      Context,
      typename ReducerDef::template Reducer<T, Context>>;
  using BackwardOp = AbstractReduceFrontGradientOp<T, Context, ReducerGradient>;
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
      if (HasArgument(def_, "num_reduce_dim")) {
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
    bool SparseFused = true>
class AbstractSortedSegmentOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AbstractSortedSegmentOp);

  bool RunOnDevice() override {
    auto& data = Input(0);
    const TIndex M = data.dim(0);
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    TIndex in_block_size = data.size() / M;
    return DispatchHelper<typename Reducer::FixedDispatch>::call(
        this, in_block_size);
  }

  template <int FixedSize>
  bool DoRunWithValue() {
    auto& data = Input(0);
    auto& segment_ids = Input(SEGMENT_IDS);
    auto* output = Output(0);

    CHECK_EQ(1, segment_ids.ndim()) << "SEGMENT_IDS must be a vector";
    TIndex N = segment_ids.dim(0);
    const TIndex M = data.dim(0);

    const TIndex* idxs;
    if (SparseFused) { // static if
      auto& indices = Input(INDICES);
      CHECK_EQ(1, indices.ndim()) << "INDICES must be a vector";
      CHECK_EQ(N, indices.dim(0))
          << "SEGMENT_IDS must have the same length as INDICES";
      idxs = indices.template data<TIndex>();
    } else {
      CHECK_EQ(N, M)
          << "DATA must have the same first dimension as SEGMENT_IDS";
    }

    // It would probably look nicer with varargs templates but it's too much
    // metaprogramming
    typename Reducer::Meta ctx;
    ctx.observeInput(0, data, 1);
    for (int i = 1; i < Reducer::kInputCount; ++i) {
      auto& aux_in = Input(i);
      CHECK_EQ(N, aux_in.dim(0))
          << "Input " << i
          << " must have have the same first dim as SEGMENT_IDS";
      ctx.observeInput(i, aux_in, 1);
    }

    const SIndex* s_ids = segment_ids.template data<SIndex>();
    const T* d = data.template data<T>();

    CHECK_GT(N, 0);
    const SIndex K = s_ids[N - 1] + 1;
    vector<TIndex> shape;
    shape.push_back(K);
    ctx.appendOutputShape(&shape);
    output->Resize(shape);

    TIndex in_block_size = data.size() / M;
    TIndex out_block_size = output->size() / K;
    T* out = output->template mutable_data<T>();

    // Assume the segments are sorted and there are no gaps
    CHECK_EQ(0, s_ids[0]) << "Indices must be sorted and not have gaps";
    for (TIndex i = 0; i < N;) {
      TIndex start = i;

      Reducer r(ctx, out + out_block_size * s_ids[start], &context_);
      for (; i < N && s_ids[start] == s_ids[i]; ++i) {
        TIndex idx;
        if (SparseFused) { // static if
          CHECK(0 <= idxs[i] && idxs[i] < M)
              << "Index out of bounds: " << idxs[i] << ", range 0 to " << M;
          idx = idxs[i];
        } else {
          idx = i;
        }
        r.template process<FixedSize>(
            ctx, d + in_block_size * idx, i, &context_);
      }

      // check correctness of the next segment
      if (i < N) {
        CHECK_EQ(s_ids[start] + 1, s_ids[i])
            << "Indices must be sorted and not have gaps";
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
};

// Gradient actually doesn't depend on whether sparse lookup is fused or not
template <typename T, typename SIndex, class Context, class ReducerGradient>
class AbstractSortedSegmentGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AbstractSortedSegmentGradientOp);

  bool RunOnDevice() override {
    auto& segment_grads = Input(SEGMENT_GRADS);
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    TIndex grad_block_size = segment_grads.size() / segment_grads.dim(0);
    return DispatchHelper<typename ReducerGradient::FixedDispatch>::call(
        this, grad_block_size);
  }

  template <int FixedSize>
  bool DoRunWithValue() {
    auto& segment_grads = Input(SEGMENT_GRADS);
    auto& segment_ids = Input(SEGMENT_IDS);
    auto* data_grads = Output(0);

    CHECK_EQ(1, segment_ids.ndim()) << "SEGMENT_IDS must be a vector";
    TIndex N = segment_ids.dim(0);

    typename ReducerGradient::Meta ctx(segment_grads, 1);
    for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
      auto& aux_in = Input(i);
      CHECK_EQ(N, aux_in.dim(0))
          << "Input " << i
          << " must have have the same first dim as SEGMENT_IDS";
      ctx.observeOriginalInput(ReducerGradient::originalInputs()[i], aux_in, 1);
    }

    const SIndex* s_ids = segment_ids.template data<SIndex>();
    const T* s_grads = segment_grads.template data<T>();

    vector<TIndex> shape;
    shape.push_back(N);
    ctx.appendGradShape(&shape);
    data_grads->Resize(shape);

    TIndex d_block_size = data_grads->size() / data_grads->dim(0);
    const SIndex K = segment_grads.dim(0);
    TIndex s_block_size = segment_grads.size() / K;
    T* out = data_grads->template mutable_data<T>();

    // Assume the segments are sorted and there are no gaps
    CHECK_EQ(0, s_ids[0]) << "Indices must be sorted and not have gaps";
    // repeat the check from forward op
    CHECK_EQ(K - 1, s_ids[N - 1]) << "Indices must be sorted and not have gaps";
    for (TIndex i = 0; i < N;) {
      TIndex start = i;

      ReducerGradient r(ctx, s_grads + s_block_size * s_ids[start], &context_);
      for (; i < N && s_ids[start] == s_ids[i]; ++i) {
        r.template fillGrad<FixedSize>(
            ctx, out + d_block_size * i, i, &context_);
      }

      // check correctness of the next segment
      if (i < N) {
        CHECK_EQ(s_ids[start] + 1, s_ids[i])
            << "Indices must be sorted and not have gaps";
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
    bool SparseFused = true>
class AbstractUnsortedSegmentOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  AbstractUnsortedSegmentOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "num_segments", num_segments_, -1) {}

  bool RunOnDevice() override {
    auto& data = Input(0);
    const TIndex M = data.dim(0);
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    TIndex in_block_size = data.size() / M;
    return DispatchHelper<typename Reducer::FixedDispatch>::call(
        this, in_block_size);
  }

  template <int FixedSize>
  bool DoRunWithValue() {
    auto& data = Input(0);
    auto& segment_ids = Input(SEGMENT_IDS);
    auto* output = Output(0);

    CHECK_EQ(1, segment_ids.ndim()) << "SEGMENT_IDS must be a vector";
    TIndex N = segment_ids.dim(0);
    const TIndex M = data.dim(0);

    const TIndex* idxs;
    if (SparseFused) { // static if
      auto& indices = Input(INDICES);
      CHECK_EQ(1, indices.ndim()) << "INDICES must be a vector";
      CHECK_EQ(N, indices.dim(0))
          << "SEGMENT_IDS must have the same length as INDICES";
      idxs = indices.template data<TIndex>();
    } else {
      CHECK_EQ(N, M)
          << "DATA must have the same first dimension as SEGMENT_IDS";
    }

    // It would probably look nicer with varargs templates but it's too much
    // metaprogramming
    typename Reducer::Meta ctx;
    ctx.observeInput(0, data, 1);
    for (int i = 1; i < Reducer::kInputCount; ++i) {
      auto& aux_in = Input(i);
      CHECK_EQ(N, aux_in.dim(0))
          << "Input " << i
          << " must have have the same first dim as SEGMENT_IDS";
      ctx.observeInput(i, aux_in, 1);
    }

    const SIndex* s_ids = segment_ids.template data<SIndex>();
    const T* d = data.template data<T>();

    // determine the number of segments
    SIndex K;
    if (num_segments_ != -1) {
      K = num_segments_;
    } else {
      K = 0;
      for (TIndex i = 0; i < N; ++i) {
        K = std::max(K, s_ids[i] + 1);
      }
    }

    vector<TIndex> shape;
    shape.push_back(K);
    ctx.appendOutputShape(&shape);
    output->Resize(shape);

    TIndex in_block_size = data.size() / M;
    TIndex out_block_size = output->size() / K;
    T* out = output->template mutable_data<T>();

    reducers_.clear();
    reducers_.reserve(K);
    for (TIndex i = 0; i < K; ++i) {
      reducers_.emplace_back(ctx, out + out_block_size * i, &context_);
    }

    for (TIndex i = 0; i < N; ++i) {
      auto s_id = s_ids[i];
      CHECK(0 <= s_id && s_id < K) << "Segment id out of range: " << s_id
                                   << ", range 0 to " << K;
      TIndex idx;
      if (SparseFused) { // static if
        CHECK(0 <= idxs[i] && idxs[i] < M) << "Index out of bounds: " << idxs[i]
                                           << ", range 0 to " << M;
        idx = idxs[i];
      } else {
        idx = i;
      }
      reducers_[s_id].template process<FixedSize>(
          ctx, d + in_block_size * idx, i, &context_);
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
  TIndex num_segments_;
  // member field to reuse memory
  vector<Reducer> reducers_;
};

// Gradient actually doesn't depend on whether sparse lookup is fused or not
template <typename T, typename SIndex, class Context, class ReducerGradient>
class AbstractUnsortedSegmentGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AbstractUnsortedSegmentGradientOp);

  bool RunOnDevice() override {
    auto& segment_grads = Input(SEGMENT_GRADS);
    // If more complicated fixed size logic becomes necessary, it can be moved
    // to the reducer class
    TIndex grad_block_size = segment_grads.size() / segment_grads.dim(0);
    return DispatchHelper<typename ReducerGradient::FixedDispatch>::call(
        this, grad_block_size);
  }

  template <int FixedSize>
  bool DoRunWithValue() {
    auto& segment_grads = Input(SEGMENT_GRADS);
    auto& segment_ids = Input(SEGMENT_IDS);
    auto* data_grads = Output(0);

    CHECK_EQ(1, segment_ids.ndim()) << "SEGMENT_IDS must be a vector";
    TIndex N = segment_ids.dim(0);

    typename ReducerGradient::Meta ctx(segment_grads, 1);
    for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
      auto& aux_in = Input(i);
      CHECK_EQ(N, aux_in.dim(0))
          << "Input " << i
          << " must have have the same first dim as SEGMENT_IDS";
      ctx.observeOriginalInput(ReducerGradient::originalInputs()[i], aux_in, 1);
    }

    const SIndex* s_ids = segment_ids.template data<SIndex>();
    const T* s_grads = segment_grads.template data<T>();

    vector<TIndex> shape;
    shape.push_back(N);
    ctx.appendGradShape(&shape);
    data_grads->Resize(shape);

    TIndex d_block_size = data_grads->size() / data_grads->dim(0);
    const SIndex K = segment_grads.dim(0);
    TIndex s_block_size = segment_grads.size() / K;
    T* out = data_grads->template mutable_data<T>();

    reducers_.clear();
    reducers_.reserve(K);
    for (SIndex i = 0; i < K; ++i) {
      reducers_.emplace_back(ctx, s_grads + s_block_size * i, &context_);
    }

    for (TIndex i = 0; i < N; ++i) {
      auto s_id = s_ids[i];
      CHECK(0 <= s_id && s_id < K) << "Segment id out of range: " << s_id
                                   << ", range 0 to " << K;
      reducers_[s_id].template fillGrad<FixedSize>(
          ctx, out + d_block_size * i, i, &context_);
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

namespace {

template <typename Def>
string FormatDoc() {
  string doc = Def::doc;
  ReplaceAll(doc, "{op}", Def::OpDef::name);
  ReplaceAll(doc, "{op_doc}", Def::OpDef::doc);
  return doc;
}

#define REGISTER_SEGMENT_DEF(...)                                              \
  REGISTER_CPU_OPERATOR_STR(                                                   \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name),              \
      __VA_ARGS__::ForwardOp);                                                 \
  OPERATOR_SCHEMA_STR(                                                         \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name))              \
      .NumInputs(__VA_ARGS__::ForwardOp::kNumInputs)                           \
      .NumOutputs(1)                                                           \
      .SetDoc(FormatDoc<__VA_ARGS__>())                                        \
      .Output(0, "OUTPUT", "Aggregated tensor")                                \
      .FillUsing(__VA_ARGS__::PopulateSchema);                                 \
  REGISTER_CPU_OPERATOR_STR(                                                   \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name) + "Gradient", \
      __VA_ARGS__::BackwardOp);                                                \
  OPERATOR_SCHEMA_STR(                                                         \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name) + "Gradient") \
      .NumInputs(__VA_ARGS__::BackwardOp::kNumInputs)                          \
      .NumOutputs(1);                                                          \
  REGISTER_GRADIENT_STR(                                                       \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name),              \
      __VA_ARGS__::GetGradient)

REGISTER_SEGMENT_DEF(
    AbstractSortedSegmentRangeDef<float, int, CPUContext, SumRangeReducerDef>);
REGISTER_SEGMENT_DEF(AbstractSortedSegmentRangeDef<
                     float,
                     int,
                     CPUContext,
                     LogSumExpRangeReducerDef>);
REGISTER_SEGMENT_DEF(AbstractSortedSegmentRangeDef<
                     float,
                     int,
                     CPUContext,
                     LogMeanExpRangeReducerDef>);
REGISTER_SEGMENT_DEF(
    AbstractSortedSegmentRangeDef<float, int, CPUContext, MeanRangeReducerDef>);
REGISTER_SEGMENT_DEF(
    AbstractSortedSegmentRangeDef<float, int, CPUContext, MaxRangeReducerDef>);

#define REGISTER_REDUCER_WITH_ALL_OPS(reducer_def)                          \
  REGISTER_SEGMENT_DEF(                                                     \
      AbstractReduceFrontDef<float, CPUContext, reducer_def>);              \
  REGISTER_SEGMENT_DEF(                                                     \
      AbstractSortedSegmentDef<float, int, CPUContext, reducer_def>);       \
  REGISTER_SEGMENT_DEF(                                                     \
      AbstractSparseSortedSegmentDef<float, int, CPUContext, reducer_def>); \
  REGISTER_SEGMENT_DEF(                                                     \
      AbstractUnsortedSegmentDef<float, int, CPUContext, reducer_def>);     \
  REGISTER_SEGMENT_DEF(                                                     \
      AbstractSparseUnsortedSegmentDef<float, int, CPUContext, reducer_def>)

REGISTER_REDUCER_WITH_ALL_OPS(SumReducerDef);
REGISTER_REDUCER_WITH_ALL_OPS(WeightedSumReducerDef);
}
}
