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

    CAFFE_CHECK_EQ(1, segment_ids.ndim()) << "SEGMENT_IDS must be a vector";
    auto N = segment_ids.dim(0);
    CAFFE_CHECK_EQ(N, data.dim(0))
        << "SEGMENT_IDS must have the same length as outer dimension of DATA";

    const SIndex* s_ids = segment_ids.template data<SIndex>();
    const T* d = data.template data<T>();

    CAFFE_CHECK_GT(N, 0);
    const SIndex K = s_ids[N - 1] + 1;
    auto shape = data.dims();
    shape[0] = K;
    output->Reshape(shape);

    TIndex block_size = data.size() / N;
    T* out = output->template mutable_data<T>();

    // Assume the segments are sorted and there are no gaps
    CAFFE_CHECK_EQ(0, s_ids[0]) << "Indices must be sorted and not have gaps";
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
        CAFFE_CHECK_EQ(s_ids[start] + 1, s_ids[i])
            << "Indices must be sorted and not have gaps";
      }
    }
    return true;
  }

  static constexpr int kNumInputs = 2;
  INPUT_TAGS(DATA, SEGMENT_IDS);
  DISABLE_COPY_AND_ASSIGN(AbstractSortedSegmentRangeOp);
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
    auto& segment_grads = Input(SEGMENT_GRADS);
    auto& segment_ids = Input(SEGMENT_IDS);
    auto* data_grads = Output(0);

    CAFFE_CHECK_EQ(1, segment_ids.ndim()) << "SEGMENT_IDS must be a vector";
    TIndex N = segment_ids.dim(0);

    const SIndex* s_ids = segment_ids.template data<SIndex>();
    const T* s_grads = segment_grads.template data<T>();

    auto shape = segment_grads.dims();
    shape[0] = N;
    data_grads->Reshape(shape);

    const SIndex K = segment_grads.dim(0);
    TIndex block_size = segment_grads.size() / K;
    T* out = data_grads->template mutable_data<T>();

    // Assume the segments are sorted and there are no gaps
    CAFFE_CHECK_EQ(0, s_ids[0]) << "Indices must be sorted and not have gaps";
    // repeat the check from forward op
    CAFFE_CHECK_EQ(K - 1, s_ids[N - 1])
        << "Indices must be sorted and not have gaps";
    for (TIndex i = 0; i < N;) {
      TIndex start = i;
      for (++i; i < N && s_ids[start] == s_ids[i]; ++i)
        ;

      RangeReducerGradient()(
          block_size,
          i - start,
          s_grads + block_size * s_ids[start],
          out + block_size * start,
          &context_);

      // check correctness of the next segment
      if (i < N) {
        CAFFE_CHECK_EQ(s_ids[start] + 1, s_ids[i])
            << "Indices must be sorted and not have gaps";
      }
    }
    return true;
  }

  static constexpr int kNumInputs = 2;
  INPUT_TAGS(SEGMENT_GRADS, SEGMENT_IDS);
  DISABLE_COPY_AND_ASSIGN(AbstractSortedSegmentRangeGradientOp);
};

template <typename T, typename SIndex, typename Context, typename ReducerDef>
struct AbstractSortedSegmentRangeDef {
  using OpDef = ReducerDef;
  static constexpr const char* basename = "SortedSegmentRange";
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
          vector<string>{GO(0), I(1)},
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
    auto* output = Output(0);

    CAFFE_CHECK_LE(num_reduce_dims_, data.ndim());

    typename Reducer::Meta ctx;
    ctx.observeInput(0, data, num_reduce_dims_);
    for (int i = 1; i < Reducer::kInputCount; ++i) {
      auto& aux_in = Input(i);
      ctx.observeInput(i, aux_in, num_reduce_dims_);
    }

    const T* d = data.template data<T>();

    vector<TIndex> shape;
    ctx.appendOutputShape(&shape);
    output->Reshape(shape);

    TIndex in_block_size = data.size_from_dim(num_reduce_dims_);
    TIndex block_num = data.size() / in_block_size;
    T* out = output->template mutable_data<T>();

    Reducer r(ctx, out, &context_);
    for (TIndex i = 0; i < block_num; ++i) {
      r.process(ctx, d + in_block_size * i, i, &context_);
    }
    return true;
  }

  static constexpr int kNumInputs = Reducer::kInputCount;
  DISABLE_COPY_AND_ASSIGN(AbstractReduceFrontOp);

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
    auto& reduction_grad = Input(REDUCTION_GRAD);
    auto& source_shape = OperatorBase::Input<vector<TIndex>>(SOURCE_SHAPE);
    auto* data_grads = Output(0);

    typename ReducerGradient::Meta ctx(reduction_grad, 0);
    for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
      auto& aux_in = Input(i);
      ctx.observeOriginalInput(
          ReducerGradient::originalInputs()[i], aux_in, num_reduce_dims_);
    }

    const T* r_grad = reduction_grad.template data<T>();

    vector<TIndex> shape(
        source_shape.begin(), source_shape.begin() + num_reduce_dims_);
    ctx.appendGradShape(&shape);
    data_grads->Reshape(shape);

    TIndex block_size = data_grads->size_from_dim(num_reduce_dims_);
    TIndex block_num = data_grads->size() / block_size;
    T* out = data_grads->template mutable_data<T>();

    ReducerGradient r(ctx, r_grad, &context_);
    for (TIndex i = 0; i < block_num; ++i) {
      r.fillGrad(ctx, out + block_size * i, i, &context_);
    }
    return true;
  }

  static constexpr int kNumInputs =
      ReducerGradient::originalInputs().size() + 2;
  enum _InputTags {
    REDUCTION_GRAD = ReducerGradient::originalInputs().size(),
    SOURCE_SHAPE
  };
  DISABLE_COPY_AND_ASSIGN(AbstractReduceFrontGradientOp);

 private:
  int num_reduce_dims_;
};

template <typename T, typename Context, typename ReducerDef>
struct AbstractReduceFrontDef {
  using OpDef = ReducerDef;
  static constexpr const char* basename = "ReduceFront";
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
              "RecordShape",
              "",
              vector<string>{I(0)},
              vector<string>{tmp_dims}),
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
    auto& segment_ids = Input(SEGMENT_IDS);
    auto* output = Output(0);

    CAFFE_CHECK_EQ(1, segment_ids.ndim()) << "SEGMENT_IDS must be a vector";
    TIndex N = segment_ids.dim(0);
    const TIndex M = data.dim(0);

    const TIndex* idxs;
    if (SparseFused) { // static if
      auto& indices = Input(INDICES);
      CAFFE_CHECK_EQ(1, indices.ndim()) << "INDICES must be a vector";
      CAFFE_CHECK_EQ(N, indices.dim(0))
          << "SEGMENT_IDS must have the same length as INDICES";
      idxs = indices.template data<TIndex>();
    } else {
      CAFFE_CHECK_EQ(N, M)
          << "DATA must have the same first dimension as SEGMENT_IDS";
    }

    // It would probably look nicer with varargs templates but it's too much
    // metaprogramming
    typename Reducer::Meta ctx;
    ctx.observeInput(0, data, 1);
    for (int i = 1; i < Reducer::kInputCount; ++i) {
      auto& aux_in = Input(i);
      CAFFE_CHECK_EQ(N, aux_in.dim(0))
          << "Input " << i
          << " must have have the same first dim as SEGMENT_IDS";
      ctx.observeInput(i, aux_in, 1);
    }

    const SIndex* s_ids = segment_ids.template data<SIndex>();
    const T* d = data.template data<T>();

    CAFFE_CHECK_GT(N, 0);
    const SIndex K = s_ids[N - 1] + 1;
    vector<TIndex> shape;
    shape.push_back(K);
    ctx.appendOutputShape(&shape);
    output->Reshape(shape);

    TIndex in_block_size = data.size() / M;
    TIndex out_block_size = output->size() / K;
    T* out = output->template mutable_data<T>();

    // Assume the segments are sorted and there are no gaps
    CAFFE_CHECK_EQ(0, s_ids[0]) << "Indices must be sorted and not have gaps";
    for (TIndex i = 0; i < N;) {
      TIndex start = i;

      Reducer r(ctx, out + out_block_size * s_ids[start], &context_);
      for (; i < N && s_ids[start] == s_ids[i]; ++i) {
        TIndex idx;
        if (SparseFused) { // static if
          CAFFE_CHECK(0 <= idxs[i] && idxs[i] < M)
              << "Index out of bounds: " << idxs[i] << ", range 0 to " << M;
          idx = idxs[i];
        } else {
          idx = i;
        }
        r.process(ctx, d + in_block_size * idx, i, &context_);
      }

      // check correctness of the next segment
      if (i < N) {
        CAFFE_CHECK_EQ(s_ids[start] + 1, s_ids[i])
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
  DISABLE_COPY_AND_ASSIGN(AbstractSortedSegmentOp);
};

// Gradient actually doesn't depend on whether sparse lookup is fused or not
template <typename T, typename SIndex, class Context, class ReducerGradient>
class AbstractSortedSegmentGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AbstractSortedSegmentGradientOp);

  bool RunOnDevice() override {
    auto& segment_grads = Input(SEGMENT_GRADS);
    auto& segment_ids = Input(SEGMENT_IDS);
    auto* data_grads = Output(0);

    CAFFE_CHECK_EQ(1, segment_ids.ndim()) << "SEGMENT_IDS must be a vector";
    TIndex N = segment_ids.dim(0);

    typename ReducerGradient::Meta ctx(segment_grads, 1);
    for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
      auto& aux_in = Input(i);
      CAFFE_CHECK_EQ(N, aux_in.dim(0))
          << "Input " << i
          << " must have have the same first dim as SEGMENT_IDS";
      ctx.observeOriginalInput(ReducerGradient::originalInputs()[i], aux_in, 1);
    }

    const SIndex* s_ids = segment_ids.template data<SIndex>();
    const T* s_grads = segment_grads.template data<T>();

    vector<TIndex> shape;
    shape.push_back(N);
    ctx.appendGradShape(&shape);
    data_grads->Reshape(shape);

    TIndex d_block_size = data_grads->size() / data_grads->dim(0);
    const SIndex K = segment_grads.dim(0);
    TIndex s_block_size = segment_grads.size() / K;
    T* out = data_grads->template mutable_data<T>();

    // Assume the segments are sorted and there are no gaps
    CAFFE_CHECK_EQ(0, s_ids[0]) << "Indices must be sorted and not have gaps";
    // repeat the check from forward op
    CAFFE_CHECK_EQ(K - 1, s_ids[N - 1])
        << "Indices must be sorted and not have gaps";
    for (TIndex i = 0; i < N;) {
      TIndex start = i;

      ReducerGradient r(
          ctx, s_grads + s_block_size * s_ids[start], &context_);
      for (; i < N && s_ids[start] == s_ids[i]; ++i) {
        r.fillGrad(ctx, out + d_block_size * i, i, &context_);
      }

      // check correctness of the next segment
      if (i < N) {
        CAFFE_CHECK_EQ(s_ids[start] + 1, s_ids[i])
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
  DISABLE_COPY_AND_ASSIGN(AbstractSortedSegmentGradientOp);
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
  using ReducerGradient =
      typename ReducerDef::template ReducerGradient<T, Context>;
  using ForwardOp = AbstractSortedSegmentOp<
      T,
      SIndex,
      Context,
      typename ReducerDef::template Reducer<T, Context>,
      false>;
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
  using ReducerGradient =
      typename ReducerDef::template ReducerGradient<T, Context>;
  using ForwardOp = AbstractSortedSegmentOp<
      T,
      SIndex,
      Context,
      typename ReducerDef::template Reducer<T, Context>>;
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
    auto& segment_ids = Input(SEGMENT_IDS);
    auto* output = Output(0);

    CAFFE_CHECK_EQ(1, segment_ids.ndim()) << "SEGMENT_IDS must be a vector";
    TIndex N = segment_ids.dim(0);
    const TIndex M = data.dim(0);

    const TIndex* idxs;
    if (SparseFused) { // static if
      auto& indices = Input(INDICES);
      CAFFE_CHECK_EQ(1, indices.ndim()) << "INDICES must be a vector";
      CAFFE_CHECK_EQ(N, indices.dim(0))
          << "SEGMENT_IDS must have the same length as INDICES";
      idxs = indices.template data<TIndex>();
    } else {
      CAFFE_CHECK_EQ(N, M)
          << "DATA must have the same first dimension as SEGMENT_IDS";
    }

    // It would probably look nicer with varargs templates but it's too much
    // metaprogramming
    typename Reducer::Meta ctx;
    ctx.observeInput(0, data, 1);
    for (int i = 1; i < Reducer::kInputCount; ++i) {
      auto& aux_in = Input(i);
      CAFFE_CHECK_EQ(N, aux_in.dim(0))
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
    output->Reshape(shape);

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
      CAFFE_CHECK(0 <= s_id && s_id < K) << "Segment id out of range: " << s_id
                                         << ", range 0 to " << K;
      TIndex idx;
      if (SparseFused) { // static if
        CAFFE_CHECK(0 <= idxs[i] && idxs[i] < M)
            << "Index out of bounds: " << idxs[i] << ", range 0 to " << M;
        idx = idxs[i];
      } else {
        idx = i;
      }
      reducers_[s_id].process(
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
  DISABLE_COPY_AND_ASSIGN(AbstractUnsortedSegmentOp);

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
    auto& segment_ids = Input(SEGMENT_IDS);
    auto* data_grads = Output(0);

    CAFFE_CHECK_EQ(1, segment_ids.ndim()) << "SEGMENT_IDS must be a vector";
    TIndex N = segment_ids.dim(0);

    typename ReducerGradient::Meta ctx(segment_grads, 1);
    for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
      auto& aux_in = Input(i);
      CAFFE_CHECK_EQ(N, aux_in.dim(0))
          << "Input " << i
          << " must have have the same first dim as SEGMENT_IDS";
      ctx.observeOriginalInput(ReducerGradient::originalInputs()[i], aux_in, 1);
    }

    const SIndex* s_ids = segment_ids.template data<SIndex>();
    const T* s_grads = segment_grads.template data<T>();

    vector<TIndex> shape;
    shape.push_back(N);
    ctx.appendGradShape(&shape);
    data_grads->Reshape(shape);

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
      CAFFE_CHECK(0 <= s_id && s_id < K) << "Segment id out of range: " << s_id
                                         << ", range 0 to " << K;
      reducers_[s_id].fillGrad(
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
  DISABLE_COPY_AND_ASSIGN(AbstractUnsortedSegmentGradientOp);

 private:
  // member field to reuse memory
  vector<ReducerGradient> reducers_;
};

template <typename T, typename SIndex, typename Context, typename ReducerDef>
struct AbstractUnsortedSegmentDef {
  using OpDef = ReducerDef;
  static constexpr const char* basename = "UnsortedSegment";
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
  using ReducerGradient =
      typename ReducerDef::template ReducerGradient<T, Context>;
  using ForwardOp = AbstractUnsortedSegmentOp<
      T,
      SIndex,
      Context,
      typename ReducerDef::template Reducer<T, Context>>;
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

#define REGISTER_SEGMENT_DEF(...)                                              \
  REGISTER_CPU_OPERATOR_STR(                                                   \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name),              \
      __VA_ARGS__::ForwardOp);                                                 \
  OPERATOR_SCHEMA_STR(                                                         \
      string(__VA_ARGS__::basename) + (__VA_ARGS__::OpDef::name))              \
      .NumInputs(__VA_ARGS__::ForwardOp::kNumInputs)                           \
      .NumOutputs(1);                                                          \
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
