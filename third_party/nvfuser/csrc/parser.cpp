#include <parser.h>

#include <arith.h>
#include <instrumentation.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <ir_iostream.h>
#include <ops/all_ops.h>
#include <type_inference.h>
#include <type_promotion.h>
#include <utils.h>

#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/csrc/jit/ir/constants.h>

#include <ATen/native/Activation.h>

#include <c10/util/CallOnce.h>

#include <unordered_map>
#include <utility>

namespace torch {
namespace jit {

typedef Value JitValue;
typedef Node JitOp;

namespace fuser {
namespace cuda {

constexpr auto kNumUnaryOps = 10;
constexpr auto kNumUnaryFloatOps = 23;
constexpr auto kNumUnaryIsOps = 6;

constexpr auto kNumBinaryFloatOps = 3;
constexpr auto kNumBinaryComparisonOps = 12;
constexpr auto kNumBinaryCastOps = 19;

constexpr auto kNumBinaryOpsWithAlpha = 6;
constexpr auto kNumLerpOps = 2;
constexpr auto kNumLayernormFwd = 2;
constexpr auto kNumBatchnormFwd = 3;
constexpr auto kNumBatchnormBwd = 2;
constexpr auto kNumInstancenormFwd = 1;
constexpr auto kNumSumToSize = 2;
constexpr auto kNumAutocastOps = 2;
constexpr auto kNumAliasDimOps = 2;
constexpr auto kNumViewOps = 2;
constexpr auto kNumVarOps = 2;
constexpr auto kNumSoftmaxFwd = 2;
constexpr auto kNumSoftmaxBwd = 2;
constexpr auto kNumAminAmaxOps = 2;

namespace {

#define REGISTER_PARSE_RULE(op, func_body, ...)                                \
  registerParseRule(                                                           \
      op,                                                                      \
      [](const Node* node, std::unordered_map<size_t, ValueHolder>& value_map) \
          -> void func_body,                                                   \
      __VA_ARGS__)

const auto& reductionSizeAttr = Symbol::attr("profiled_reduction_size");
const auto& viewSizeAttr = Symbol::attr("profiled_view_size");
const auto& intListAttr = Symbol::attr("profiled_int_list");
const auto& intAttr = Symbol::attr("profiled_int");
const auto& boolListAttr = Symbol::attr("profiled_bool_list");
const auto& boolAttr = Symbol::attr("profiled_bool");
const auto& strAttr = Symbol::attr("profiled_str");
const auto& ivalAttr = Symbol::attr("profiled_ival");
const auto& profileFailedAttr = Symbol::attr("profile_failed");

typedef Val* CgValue;
typedef Expr* CgOp;

Val* castTensoToDtype(CgValue self, JitValue* cast_val) {
  auto cast_ival = toIValue(cast_val);
  // we need static type for cast
  TORCH_INTERNAL_ASSERT(cast_ival.has_value());
  if (cast_ival->isInt()) {
    auto dtype = cast_ival->toScalarType();

    // We want to keep our internal fusion math in FP32
    // Shape Inference will continue to propagate the right
    // type to outputs unchanged.
    if (dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16) {
      dtype = at::ScalarType::Float;
    }

    return castOp(aten_to_data_type(dtype), self);
  } else {
    TORCH_INTERNAL_ASSERT(
        cast_ival->isNone(),
        "unrecognized dtype option, expect 'int' but got: ",
        cast_ival->tagKind());

    // return a copy if dtype is `None`
    return set(self);
  }
}

bool isReductionNonCompatibleTensor(
    const std::shared_ptr<c10::TensorType>& tensor_type) {
  return is_zero_dim_tensor(tensor_type) || is_zero_sized_tensor(tensor_type);
}

bool isInputNonSizeZeroTensor(const Node* node) {
  for (const auto& val : node->inputs()) {
    auto tensor_type = val->type()->cast<TensorType>();
    if (tensor_type && is_zero_sized_tensor(tensor_type)) {
      return false;
    }
  }
  return true;
}

bool isScalarTypeCompatible(const Node* node, size_t offset) {
  auto val = node->input(offset);
  // return true if it's not specified
  if (val->type()->isSubtypeOf(static_cast<c10::TypePtr>(NoneType::get()))) {
    return true;
  }
  // return false if it's runtime value
  if (val->node()->kind() != prim::Constant) {
    return false;
  }
  auto dtype = toIValue(val)->toScalarType();

  // we do NOT support half math type yet
  if (dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16) {
    return false;
  }
  return true;
}

// Note [ Permutation Bookkeeping and Propagation in Parser ]
//
// The goal in supporting permutation propagation in parser is to:
//   1. resolves conflicts and propagate permutation;
//   2. bookkeeping of permutation on existing tensors;
//
// The requirement right now is that all parsing rules should support
// non-permuted inputs, some binary operations support inputs with arbitrary
// permutation, a few operations support special inputs.
// In case where "wrong" inputs are fed to an operation, we should transpose
// it to proper supported permutation. This allows us to progressively expand
// permutation support.
// Currently we bind all permuted codegen Val in `ValueHolder`. This saves
// unnecessary transpose (not sure if it actually helps) since we can reuse
// permuted tensors.
//
// Parsing rule pattern:
// a. ops that only support non-permuted inputs (e.g. sum)
//
//    // Specifying `MemoryFormat::Contiguous` here to force all inputs to be in
//    // `Contiguous`
//    auto [format, self] = getConsistentValues(
//        MemoryFormat::Contiguous,
//        value_map[node->inputs()[0]->unique()]);
//    // ... use self
//
// b. format agnostic ops (e.g. PW unary/binary op like aten::add)
//
//    // getConsistentValues -> return target format and copies of operands in
//    // the same format
//    auto [format, lhs, rhs] = getConsistentValues(
//        c10::nullopt,
//        value_map[node->inputs()[0]->unique()],
//        value_map[node->inputs()[1]->unique()]);
//
//    // compute out
//    auto out = binaryOp(op_mapping[node->kind()], lhs, rhs);
//    // specify `format` for out when adding it to `value_map_`
//    value_map.emplace(node->output()->unique(), ValueHolder(out, format));
//
// c. ops that supports special permutation. e.g. aten::batch_norm with
//    channels-last inputs.

struct MemoryFormat {
  // indices of dimensions with increasing stride.
  std::vector<int> permuted_order_;

  // permutation_ encodes `permuted_order_` by concatenating all elements, with
  // the exception for unpermuted tensor, where we special case permutation_ to
  // be 0.
  //
  // e.g. for an channels-last tensor, permutation_ would be (n-1)123...(n-2);
  // Note: we are omitting the leading '0' when applicable, and apparently this
  //       encoding only works with rank < 10
  // see [ Note: MemoryFormat and Stride Order ]
  size_t permutation_ = 0;

  // default to non-permuted tensor
  MemoryFormat() = default;

  // [ Note: MemoryFormat and Stride Order ]
  // stride_order is extracted from
  //     `TensorType::stride_properties()::stride_index_`, it describes the
  // index of axes from fastest to slowest.
  // or a 4d tensor, if we have stride_order = {x0, x1, x2, x3}, The i-th
  // fastest dimension would be stride_order[i].
  //
  // Look at comment for c10::Stride in aten/src/ATen/core/jit_type.h
  //
  // eg0. for rank 4 non-permuted tensor, stride_order would be {3, 2, 1, 0}, it
  // means the fastest dimension is axis-3. the next one would be 2, e.t.c.. So
  // it's a non-permuted tensor.
  //      it should be encoded as permutation_ = 3210 (we special case it to 0)
  //
  // eg1. for rank 4 channels-last tensor, stride_order would be {1, 3, 2, 0},
  // it means the fastest dimension is axis-1. the next one would be 3, and then
  // 2, and then 0. So this is a channels last tensor (NCHW).
  //      it will be encoded as permutation_ = 1320
  //
  // eg2. for a rank 4 permuted tensor, stride_order can be {0, 3, 2, 1}
  //      it will be encoded as permutation_ = 321 (omitting leading '0')
  void setPermutation(const std::vector<int>& stride_order) {
    int rank = stride_order.size();
    TORCH_INTERNAL_ASSERT(
        rank <= 10, "MemoryFormat for permutation only supports rank <= 10");

    // storing stride_order in `permuted_order` for a simpler life, so we don't
    // have to decode `permutation_` when we want to apply/restore permutation_.
    permuted_order_ = stride_order;
    bool has_permutation = false;
    permutation_ = 0;
    for (const auto i : c10::irange(rank)) {
      permutation_ = permutation_ * 10 + stride_order[i];
      if (!has_permutation && stride_order[i] != rank - 1 - i) {
        has_permutation = true;
      }
    }

    // special case permutation_ to reflect non-permuted tensor
    if (!has_permutation) {
      permutation_ = 0;
    }
  }

  // returns the stride order for given MemoryFormat encoding permutation_
  //
  // see details for encoding in [ Note: MemoryFormat and Stride Order ]
  std::vector<int> toStrideOrder() const {
    std::vector<int> stride_order;
    // return empty vector for no permutation
    if (hasPermutation()) {
      // be generous with reserved space
      stride_order.reserve(10);
      bool encountered_zero = false;
      size_t permutation = permutation_;
      while (permutation != 0) {
        int order = static_cast<int>(permutation % 10);
        permutation /= 10;
        if (order == 0) {
          encountered_zero = true;
        }
        stride_order.push_back(order);
      }
      if (!encountered_zero) {
        // in case leading '0' is omitted, push it back
        stride_order.push_back(0);
      }
      // since we use push_back, our stride_order is reversed.
      std::reverse(stride_order.begin(), stride_order.end());
    }
    return stride_order;
  }

  // returns c10::nullopt when it's not safe to broadcast current permutation to
  // rank
  c10::optional<MemoryFormat> broadcastToRank(size_t rank) const {
    auto ret = Contiguous();
    if (hasPermutation()) {
      auto stride_order = toStrideOrder();
      auto cur_rank = stride_order.size();
      // no op for (cur_rank == 0) || (cur_rank == rank)
      if (cur_rank < rank) {
        // broadcasting to hight rank can be done by:
        //   1. incrementing all existing stride order by rank_diff;
        //   2. push back decrementing elements starting with rank_diff;
        //   where rank_diff = rank - cur_rank
        //
        // see [ Note: MemoryFormat and Stride Order]
        // e.g.
        //   taking broadcasted bias for channels last as an example
        //     stride_order = {0, 2, 1} broadcasted to rank == 4 would give us
        //     rank_diff = 4 - 3 = 1
        //     take step 1 -> {1, 3, 2}
        //     take step 2 -> {1, 3, 2, 0}
        int rank_diff = static_cast<int>(rank - cur_rank);
        for (auto& val : stride_order) {
          val += rank_diff;
        }
        for (int i = rank_diff - 1; i >= 0; i--) {
          stride_order.push_back(i);
        }
      } else if (cur_rank > rank) {
        // shrink permutation to lower rank. We can simply discard higher rank
        // stride order when they are not permuted to lower rank bit, because in
        // those instance we can't obey broadcasting semantics while preserving
        // permutation. We check for stride order and ensure that the lower
        // `rank` bits are all permuted within the lower rank. Afterwards, we
        // update stride_order by decrement each entry by rank_diff to reflect
        // correct stride order.
        //
        // see [ Note: MemoryFormat and Stride Order]
        // e.g. for rank 4 channels last {1, 3, 2, 0}:
        //   1. format can safely shrink to rank 3, since any@{1, 3, 2} >=
        //   (4-3); We ditch last (4-3) rank and decrement each element by (4-1)
        //   that gives us {0, 2, 1};
        //   2. but when we shrink it to rank 2, we have {1, 3} where 1 < (4-2)
        //   and it can't be handled, we return c10::nullopt.
        int collapsed_ranks = static_cast<int>(cur_rank - rank);
        for (size_t i = 0; i < rank; i++) {
          if (stride_order[i] < collapsed_ranks) {
            // illegal collapsing, return c10::nullopt
            return c10::nullopt;
          }
          // update collapsed stride_order
          stride_order[i] -= collapsed_ranks;
        }
        // discard higher rank stride order.
        stride_order.resize(rank);
      }
      ret.setPermutation(stride_order);
    }
    return ret;
  }

  // returns non-permuted format
  static MemoryFormat Contiguous() {
    return MemoryFormat();
  }

  bool hasPermutation() const {
    return permutation_ != 0;
  }

  bool isChannelsLast() const {
    int rank = permuted_order_.size();

    if (rank > 2 && permuted_order_[0] == 1 && permuted_order_[rank - 1] == 0) {
      for (const auto i : c10::irange(rank - 2)) {
        if (permuted_order_[i + 1] != rank - 1 - i) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  // returns transpose map to achieve permutation on non-permuted tensor
  // note: used for aten::permute API and codegen tranpose API
  std::vector<int64_t> apply() const {
    std::vector<int64_t> ret;
    if (hasPermutation()) {
      ret.resize(permuted_order_.size());
      std::copy(permuted_order_.rbegin(), permuted_order_.rend(), ret.begin());
    }
    return ret;
  }

  // returns transpose map to restore back to non-permuted tensor
  // note: used for aten::permute API and codegen transpose API
  std::vector<int64_t> restore() const {
    std::vector<int64_t> ret;
    if (hasPermutation()) {
      int rank = permuted_order_.size();
      ret.resize(rank);
      for (const auto i : c10::irange(rank)) {
        ret[permuted_order_[i]] = rank - 1 - i;
      }
    }
    return ret;
  }
};

struct MemoryCompare {
  bool operator()(const MemoryFormat& format0, const MemoryFormat& format1)
      const {
    return format0.permutation_ < format1.permutation_;
  }
};

typedef std::map<MemoryFormat, CgValue, MemoryCompare> MemoryFormatMap;

MemoryFormat operator+(const MemoryFormat& a, const MemoryFormat& b) {
  // Note: TensorIterator logic uses first input to dominate output MemoryFormat
  // so instead of `a.permutation_ >= b.permutation_ ? a : b;`, we use:
  return a;
};

//! ValueHolder is holds multiple copies in different permutation `MemoryFormat`
//! of a tensor view. This mainly serves two purposes:
//!
//!   1. reuse permuted tensor views among consumers
//!   2. bookkeeping for permuted tensor views in input/output tensors
//!
//! refer to  Note [ Permutation Bookkeeping and Propagation in Parser ]
class ValueHolder {
 public:
  // checks if given Val in target format exists.
  bool hasValue(const MemoryFormat& format) const {
    return vals_.count(format) != 0;
  }

  // returns Val in target format.
  CgValue value(const MemoryFormat& format) const {
    auto iter_val = vals_.find(format);
    TORCH_INTERNAL_ASSERT(
        iter_val != vals_.end(), "accessing non existing c_last_value()");
    return iter_val->second;
  }

  // returns Val in target format if it exists, otherwise, transpose an existing
  // copy and add that to bookkeeping.
  CgValue maybeConvertValue(const MemoryFormat& format) {
    auto cur_rank = rank();
    // scalar (tensor) where cur_rank == 0, memory format doesn't carry meaning
    // and should just return the value as-is. same for non-tensor where
    // cur_rank == -1
    if (cur_rank <= 0) {
      return std::get<1>(getEntry());
    }
    MemoryFormat format_s;
    CgValue value_s = nullptr;
    std::tie(format_s, value_s) = getEntry();

    auto opt_format_d = format.broadcastToRank(static_cast<size_t>(cur_rank));
    TORCH_INTERNAL_ASSERT(
        opt_format_d.has_value(),
        "maybeConvertValue requested for illegal permutation");
    MemoryFormat format_d = opt_format_d.value();

    auto iter_val = vals_.find(format_d);
    if (iter_val != vals_.end()) {
      return iter_val->second;
    }
    auto val = convertValue(format_d, format_s, value_s);
    vals_[format_d] = val;
    return val;
  }

  int rank() const {
    if (!is_tensor_view_) {
      return -1;
    } else {
      auto v = std::get<1>(getEntry());
      TORCH_INTERNAL_ASSERT(
          v->isA<TensorView>(), "can only access rank of TensorView");
      return static_cast<int>(v->as<TensorView>()->nDims());
    }
  }

  // TODO: delete this and update accessor for value_map(_)
  ValueHolder() {
    TORCH_INTERNAL_ASSERT(false, "can't default constructor ValueHolder");
  }

  ValueHolder(CgValue val, MemoryFormat format = MemoryFormat()) {
    vals_[format] = val;
    if (val->isA<TensorView>()) {
      is_tensor_view_ = true;
    }
  }

  // returns the MemoryFormat and codegen Val with the highest precedence among
  // existing copies.
  std::tuple<MemoryFormat, CgValue> getEntry() const {
    TORCH_CHECK(!vals_.empty(), "ValueHolder::getEntry() on empty vals_");
    // return the last entry, this allows us to prioritize permuted (e.g.
    // channels-last) tensor over non-permuted tensors
    return *vals_.rbegin();
  }

  // TODO: code cleaning in parser so we don't need these.
  // returns Val*, keeping them here just so we have less code change.
  CgValue operator*() const {
    return std::get<1>(getEntry());
  }
  CgValue operator->() const {
    return std::get<1>(getEntry());
  }
  operator CgValue() const {
    return std::get<1>(getEntry());
  }

 private:
  // helper function to convert value_s @ format_s to format_d
  CgValue convertValue(
      MemoryFormat format_d,
      MemoryFormat format_s,
      CgValue value_s) {
    TORCH_INTERNAL_ASSERT(
        value_s->isA<TensorView>(), "cannot convert non-TensorView");
    auto tv = value_s->as<TensorView>();
    // TODO: we could probably merge the two if it has perf impact on generated
    // kernel

    // restore source permutation
    if (format_s.hasPermutation()) {
      tv = permute(tv, format_s.restore());
    }
    // apply destination permutation
    if (format_d.hasPermutation()) {
      tv = permute(tv, format_d.apply());
    }
    return tv;
  }

 private:
  // container to hold all copies of value in different MemoryFormat
  // std::unordered_map<MemoryFormat, CgValue> vals_;
  MemoryFormatMap vals_;

  // identify scalar Val
  bool is_tensor_view_ = false;
};

template <class Func, class... Values>
auto iterate(Func f, ValueHolder& val) {
  return f(val);
}

template <class Func, class... Values>
auto iterate(Func f, ValueHolder& val, Values&... vals) {
  return f(val, iterate(f, vals...));
}

// iterate through all vals and return the output MemoryFormat and copies of
// vals.
//   1. When `forced_format == c10::nullopt`, target MemoryFormat returns the
//      format of the first val in `vals`, this is to achieve a coherent
//      behavior as with eager TensorIterator;
//   2. The target can be overwritten vias specifying `forced_format`.
//
// Note: take `Values&` by reference, since `maybeConvertValue` needs to modify
// the entry and we want that to be updated in `value_map_`
template <class... Values>
std::pair<MemoryFormat, std::list<CgValue>> getConsistentValues(
    c10::optional<MemoryFormat> forced_format,
    Values&... vals) {
  MemoryFormat format;
  if (forced_format.has_value()) {
    format = forced_format.value();
  } else {
    // check for identical nDim on vals
    auto rank_func = [](const ValueHolder& val, int rank = 0) {
      int v_rank = val.rank();
      v_rank = std::max(0, v_rank);
      if (rank == 0) {
        return v_rank;
      } else if (v_rank == 0) {
        return rank;
      } else if (rank == -1 || v_rank != rank) {
        return -1;
      }
      return rank;
    };
    int rank = iterate(rank_func, vals...);

    // TODO: this is not needed as we are only using the first val
    // only apply permutation when all inputs are of identical rank, since
    // permutation could have changed semantics among broadcasted tensors.
    // Consider pointwise operation between two tensor [N, C, H, W] + [H, W]
    if (rank > 0) {
      auto format_func = [](const ValueHolder& val,
                            MemoryFormat f = MemoryFormat::Contiguous()) {
        return std::get<0>(val.getEntry()) + f;
      };
      format = iterate(format_func, vals...);
    } else {
      format = MemoryFormat::Contiguous();
    }
  }

  auto convert_func = [format](
                          ValueHolder& val, std::list<CgValue> list_val = {}) {
    list_val.push_front(val.maybeConvertValue(format));
    return list_val;
  };
  auto list_val = iterate(convert_func, vals...);

  return std::make_pair(format, list_val);
}

// iterate through all vals and return the output MemoryFormat and copies of
// vals.
//   1. When `forced_format == c10::nullopt`, target MemoryFormat returns the
//      format of the first val in `vals`, this is to achieve a coherent
//      behavior as with eager TensorIterator;
//   2. The target can be overwritten vias specifying `forced_format`.
//
// Note: take `Values&` by reference, since `maybeConvertValue` needs to modify
// the entry and we want that to be updated in `value_map_`
template <class... Values>
std::pair<MemoryFormat, std::list<CgValue>> getPWFormatValues(
    c10::optional<MemoryFormat> forced_format,
    Values&... vals) {
  MemoryFormat format;
  if (forced_format.has_value()) {
    format = forced_format.value();
  } else {
    // get maximum rank on vals
    std::vector<MemoryFormat> formats;
    std::vector<int> ranks;
    auto max_rank_func = [&ranks](const ValueHolder& val, int rank = 0) {
      int v_rank = val.rank();
      ranks.push_back(v_rank);
      return std::max(rank, v_rank);
    };
    int max_rank = iterate(max_rank_func, vals...);

    // going through all permutation, keeping consistency with TensorIterator
    // behavior and the first tensor with highest rank dictates output
    // permutation
    auto format_func = [&formats, &max_rank](
                           const ValueHolder& val,
                           MemoryFormat f = MemoryFormat::Contiguous()) {
      auto cur_format = std::get<0>(val.getEntry());
      formats.push_back(cur_format);
      return val.rank() == max_rank ? cur_format : f;
    };
    format = iterate(format_func, vals...);

    // we need to do pair-wise comparison to ensure that all permutation are
    // compatible since permutation could have changed semantics among
    // broadcasted tensors. Consider pointwise operation between three tensor
    // [N, C, H, W] + [C, H, W] + [H, W]
    for (size_t i = 0; i < formats.size() && format.hasPermutation(); i++) {
      for (size_t j = 0; j < formats.size(); j++) {
        // don't compare scalar tensor or scalar
        if (ranks[i] <= 0 || ranks[j] <= 0 || i == j) {
          continue;
        }
        size_t lower_rank = std::min(ranks[i], ranks[j]);
        auto i_format = formats[i].broadcastToRank(lower_rank);
        auto j_format = formats[j].broadcastToRank(lower_rank);

        // breaks permutation if any:
        //   1. i_format can't be broadcasted to lower_rank;
        //   2. j_format can't be broadcasted to lower_rank;
        if (!i_format.has_value() || !j_format.has_value()) {
          format = MemoryFormat::Contiguous();
        }
      }
    }
  }

  auto convert_func = [format](
                          ValueHolder& val, std::list<CgValue> list_val = {}) {
    list_val.push_front(val.maybeConvertValue(format));
    return list_val;
  };
  auto list_val = iterate(convert_func, vals...);

  return std::make_pair(format, list_val);
}

typedef void (
    *ParseFuncPtr)(const Node*, std::unordered_map<size_t, ValueHolder>&);
typedef bool (*MergeQueryFuncPtr)(const Node*);

// TODO: add a mutex to make it thread safe.
class IrParser {
  enum class OperatorType {
    ElementWise,
    Reduction,
    ReductionToSize,
    Normalization
  };
  typedef OperatorType (*OperatorTypeFuncPtr)(const Node*);

  class RegistrationEntry {
   public:
    RegistrationEntry(
        ParseFuncPtr parse_f,
        MergeQueryFuncPtr merge_f = nullptr,
        OperatorTypeFuncPtr type_f = nullptr)
        : parse_f_(parse_f), merge_f_(merge_f), type_f_(type_f) {}

    void parse(
        const Node* node,
        std::unordered_map<size_t, ValueHolder>& values) const {
      parse_f_(node, values);
    }

    bool isCompatible(const Node* node) const {
      if (merge_f_ == nullptr) {
        return true;
      }
      return merge_f_(node);
    }

    bool isType(const Node* node, OperatorType type) const {
      auto n_type =
          type_f_ == nullptr ? OperatorType::ElementWise : type_f_(node);
      return n_type == type;
    }

   private:
    ParseFuncPtr parse_f_;
    MergeQueryFuncPtr merge_f_;
    OperatorTypeFuncPtr type_f_;
  };

 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  IrParser(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {
    initRegistry();
  }

  std::unique_ptr<Fusion> parse() {
    auto fusion = std::make_unique<Fusion>();
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    FusionGuard fg(fusion.get());
    auto block = graph_->block();

    std::unordered_map<Val*, MemoryFormat> permuted_tensors;
    // register all inputs;
    for (auto val : block->inputs()) {
      TORCH_INTERNAL_ASSERT(
          registerValue(val),
          "Failure when register value: ",
          *(val->node()),
          " with type: ",
          val->type()->repr_str());
      MemoryFormat format;
      Val* operand = nullptr;
      std::tie(format, operand) = value_map_[val->unique()].getEntry();
      fusion->addInput(operand);

      // mark input tensor as permuted;
      if (format.hasPermutation()) {
        permuted_tensors.insert({operand, format});
      }

      auto opt_dtype = operand->getDataType();
      // computation promotion, we cast fp16 or bf16 inputs to fp32 and use
      // promoted type in the computation.
      if (opt_dtype.has_value() &&
          (opt_dtype.value() == DataType::Half ||
           opt_dtype.value() == DataType::BFloat16)) {
        Val* promoted_val = castOp(DataType::Float, operand);
        value_map_[val->unique()] = ValueHolder(promoted_val, format);
      }
    }

    // compose nodes in topo order;
    for (const JitOp* node : block->nodes()) {
      processJitNode(node);
    }

    // mark output;
    for (auto jit_output : block->outputs()) {
      MemoryFormat format;
      Val* operand = nullptr;
      std::tie(format, operand) = value_map_[jit_output->unique()].getEntry();
      TensorView* out = operand->as<TensorView>();
      // demote output dtype to be match PyTorch JIT graph.
      auto tensor_type = jit_output->type()->cast<TensorType>();
      TORCH_INTERNAL_ASSERT(
          tensor_type, "output of fusion group is not TensorType.");
      if (tensor_type->scalarType().has_value()) {
        out = optionalCastStrict(
                  aten_to_data_type(*tensor_type->scalarType()), out)
                  ->as<TensorView>();
      }

      if (out->isFusionOutput()) {
        // TODO: This is wasted memory bandwidth, we need to copy since we can't
        // output a tensor twice.
        out = set(out);
      }

      fusion->addOutput(out);

      // mark output tensor as permuted;
      if (format.hasPermutation()) {
        permuted_tensors.insert({out, format});
      }
    }

    for (const auto& i : c10::irange(fusion->inputs().size())) {
      const auto& entry = permuted_tensors.find(fusion->inputs()[i]);
      if (entry != permuted_tensors.end()) {
        fusion->setPermutationOnInput(i, entry->second.apply());
      }
    }
    for (const auto& i : c10::irange(fusion->outputs().size())) {
      const auto& entry = permuted_tensors.find(fusion->outputs()[i]);
      if (entry != permuted_tensors.end()) {
        fusion->setPermutationOnOutput(i, entry->second.restore());
      }
    }
    return fusion;
  }

  static bool lookupInSymbolSet(const Node* node) {
    initRegistry();

    std::lock_guard<std::mutex> lock(parser_mutex_);
    return parser_symbol_set_.count(node->kind()) != 0;
  }

  // return nullptr if entry does not exist
  static const RegistrationEntry* lookupInRegistry(const Node* node) {
    std::lock_guard<std::mutex> lock(parser_mutex_);

    if (parser_skip_set_.count(node->kind()) != 0) {
      return nullptr;
    }
    // we need to use maybeSchema for nodes like prim::Constant, which doesn't
    // have a schema
    auto schema_ptr = node->maybeSchema();
    if (schema_ptr != nullptr) {
      // search cached entry first
      auto cache_it = cached_registry_lookup_.find(schema_ptr);
      if (cache_it != cached_registry_lookup_.end()) {
        return cache_it->second;
      } else {
        // match signature
        auto schema_str = canonicalSchemaString(*schema_ptr);

        auto iter = jit_operator_registry_.find(schema_str);
        if (iter != jit_operator_registry_.end()) {
          // update cache entry
          cached_registry_lookup_.insert(cache_it, {schema_ptr, &iter->second});
          return &iter->second;
        }
      }
    }
    return nullptr;
  }

  static bool querySkipSymbolSet(c10::Symbol symbol, bool flip) {
    initRegistry();

    std::lock_guard<std::mutex> lock(parser_mutex_);
    // no need to init registry here (unlike `lookupInSymbolSet`, as
    // `parser_skip_set_` is not initialized via initialization
    bool ret = parser_skip_set_.count(symbol) != 0;
    if (flip) {
      if (ret) {
        parser_skip_set_.erase(symbol);
      } else {
        parser_skip_set_.insert(symbol);
      }
    }
    return ret;
  }

  static void initRegistry() {
    c10::call_once(once_flag_, []() {
      std::lock_guard<std::mutex> lock(parser_mutex_);
      registerJitOperator();
    });
  }

  static bool canParseNode(const Node* node) {
    initRegistry();

    // match signature.
    auto schema_ptr = node->maybeSchema();
    if (schema_ptr == nullptr) {
      return false;
    }
    auto reg_entry = lookupInRegistry(node);
    return reg_entry != nullptr && reg_entry->isCompatible(node);
  }

  static bool isReductionToSizeNode(const Node* node) {
    initRegistry();

    auto reg_entry = lookupInRegistry(node);
    return reg_entry != nullptr &&
        reg_entry->isType(node, OperatorType::ReductionToSize);
  }

  static bool isReductionNode(const Node* node) {
    initRegistry();

    auto reg_entry = lookupInRegistry(node);
    return reg_entry != nullptr &&
        (reg_entry->isType(node, OperatorType::Reduction) ||
         reg_entry->isType(node, OperatorType::ReductionToSize));
  }

  static bool isNormalizationNode(const Node* node) {
    initRegistry();

    auto reg_entry = lookupInRegistry(node);
    return reg_entry != nullptr &&
        reg_entry->isType(node, OperatorType::Normalization);
  }

  static bool isElementWiseNode(const Node* node) {
    initRegistry();

    auto reg_entry = lookupInRegistry(node);
    return reg_entry != nullptr &&
        reg_entry->isType(node, OperatorType::ElementWise);
  }

  // TODO: is_reduction is too hacky here. we should categorize operation types
  //       based on their memory accessing pattern, which would affect fusion
  //       strategy and partition logic.
  static void registerParseRule(
      std::shared_ptr<Operator>& op,
      ParseFuncPtr parse_fn,
      MergeQueryFuncPtr merge_query_fn = nullptr,
      OperatorTypeFuncPtr type_fn = nullptr) {
    auto op_name = op->schema().name();
    parser_symbol_set_.insert(c10::Symbol::fromQualString(op_name));
    // We blindly attempt to profile the inplace version of supported op, this
    // is to ensure that in-place removal in fusion partition would have the
    // profile information for them readily available after the pass.
    parser_symbol_set_.insert(c10::Symbol::fromQualString(op_name + '_'));
    jit_operator_registry_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(canonicalSchemaString(op->schema())),
        std::forward_as_tuple(parse_fn, merge_query_fn, type_fn));
  }

 private:
  static void registerJitOperator() {
    // Register parse-function for each JIT operator;
    // This is a one-time look up, our hash registry indexes on the pointer in
    // OperatorRegistry.

    std::array<const char*, kNumBinaryOpsWithAlpha> BinaryOpWithAlpha = {
        "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
        "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor",
        "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
        "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor",
        "aten::rsub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
        "aten::rsub(Tensor self, Scalar other, Scalar alpha) -> Tensor"};
    for (auto signature : BinaryOpWithAlpha) {
      auto ptr_op = getOperatorForLiteral(signature);
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            using BinaryOpWithAlphaType = Val* (*)(Val*, Val*, Val*);
            static std::unordered_map<
                Symbol,
                std::pair<BinaryOpType, BinaryOpWithAlphaType>>
                op_mapping(
                    {{aten::add,
                      std::make_pair(
                          BinaryOpType::Add,
                          static_cast<BinaryOpWithAlphaType>(&add_alpha))},
                     {aten::sub,
                      std::make_pair(
                          BinaryOpType::Sub,
                          static_cast<BinaryOpWithAlphaType>(&sub_alpha))},
                     {aten::rsub,
                      std::make_pair(
                          BinaryOpType::Sub,
                          static_cast<BinaryOpWithAlphaType>(&sub_alpha))}});
            // TODO: handle scaling factor when it's not constant 1;
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getPWFormatValues(
                c10::nullopt,
                value_map[node->inputs()[0]->unique()],
                value_map[node->inputs()[1]->unique()]);
            auto lhs = list_val.front();
            list_val.pop_front();
            auto rhs = list_val.front();
            list_val.pop_front();
            Val* alpha = value_map[node->inputs()[2]->unique()];

            auto out = alpha->isOneInt()
                ? binaryOp(
                      op_mapping[node->kind()].first,
                      node->kind() == aten::rsub ? rhs : lhs,
                      node->kind() == aten::rsub ? lhs : rhs,
                      TypePromotion::default_op_config)
                : (node->kind() == aten::rsub
                       ? op_mapping[node->kind()].second(rhs, lhs, alpha)
                       : op_mapping[node->kind()].second(lhs, rhs, alpha));
            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    std::array<const char*, kNumBinaryFloatOps> BinaryFloatOp = {
        "aten::div(Tensor self, Tensor other) -> Tensor",
        "aten::div(Tensor self, Scalar other) -> Tensor",
        "aten::atan2(Tensor self, Tensor other) -> Tensor"};
    for (auto signature : BinaryFloatOp) {
      auto ptr_op = getOperatorForLiteral(signature);
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            static std::unordered_map<Symbol, BinaryOpType> op_mapping(
                {{aten::div, BinaryOpType::Div},
                 {aten::atan2, BinaryOpType::Atan2}});

            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getPWFormatValues(
                c10::nullopt,
                value_map[node->inputs()[0]->unique()],
                value_map[node->inputs()[1]->unique()]);
            auto lhs = list_val.front();
            list_val.pop_front();
            auto rhs = list_val.front();
            list_val.pop_front();

            auto out = binaryOp(
                op_mapping[node->kind()],
                lhs,
                rhs,
                TypePromotion::float_op_config);
            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    std::array<const char*, kNumBinaryCastOps> BinaryCastOp = {
        "aten::mul(Tensor self, Tensor other) -> Tensor",
        "aten::mul(Tensor self, Scalar other) -> Tensor",
        "aten::max(Tensor self, Tensor other) -> Tensor",
        "aten::min(Tensor self, Tensor other) -> Tensor",
        "aten::pow(Tensor self, Tensor exponent) -> Tensor",
        "aten::pow(Tensor self, Scalar exponent) -> Tensor",
        "aten::pow(Scalar self, Tensor exponent) -> Tensor",
        "aten::remainder(Tensor self, Tensor other) -> Tensor",
        "aten::fmod(Tensor self, Tensor other) -> Tensor",
        "aten::bitwise_and(Tensor self, Tensor other) -> Tensor",
        "aten::__and__(Tensor self, Tensor other) -> Tensor",
        "aten::bitwise_or(Tensor self, Tensor other) -> Tensor",
        "aten::__or__(Tensor self, Tensor other) -> Tensor",
        "aten::bitwise_xor(Tensor self, Tensor other) -> Tensor",
        "aten::__xor__(Tensor self, Tensor other) -> Tensor",
        "aten::bitwise_left_shift(Tensor self, Tensor other) -> Tensor",
        "aten::__lshift__(Tensor self, Tensor other) -> Tensor",
        "aten::bitwise_right_shift(Tensor self, Tensor other) -> Tensor",
        "aten::__rshift__(Tensor self, Tensor other) -> Tensor"};
    for (auto signature : BinaryCastOp) {
      auto ptr_op = getOperatorForLiteral(signature);
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            static std::unordered_map<Symbol, BinaryOpType> op_mapping(
                {{aten::mul, BinaryOpType::Mul},
                 {aten::min, BinaryOpType::Min},
                 {aten::max, BinaryOpType::Max},
                 {aten::pow, BinaryOpType::Pow},
                 {aten::remainder, BinaryOpType::Remainder},
                 {aten::fmod, BinaryOpType::Fmod},
                 {aten::bitwise_and, BinaryOpType::And},
                 {aten::__and__, BinaryOpType::And},
                 {aten::bitwise_or, BinaryOpType::Or},
                 {aten::__or__, BinaryOpType::Or},
                 {aten::bitwise_xor, BinaryOpType::Xor},
                 {aten::__xor__, BinaryOpType::Xor},
                 {aten::bitwise_left_shift, BinaryOpType::Lshift},
                 {aten::__lshift__, BinaryOpType::Lshift},
                 {aten::bitwise_right_shift, BinaryOpType::Rshift},
                 {aten::__rshift__, BinaryOpType::Rshift}});

            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getPWFormatValues(
                c10::nullopt,
                value_map[node->inputs()[0]->unique()],
                value_map[node->inputs()[1]->unique()]);
            auto lhs = list_val.front();
            list_val.pop_front();
            auto rhs = list_val.front();
            list_val.pop_front();

            auto out = binaryOp(
                op_mapping[node->kind()],
                lhs,
                rhs,
                TypePromotion::default_op_config);
            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    std::array<const char*, kNumBinaryComparisonOps> BinaryOp = {
        "aten::eq(Tensor self, Tensor other) -> Tensor",
        "aten::eq(Tensor self, Scalar other) -> Tensor",
        "aten::ne(Tensor self, Tensor other) -> Tensor",
        "aten::ne(Tensor self, Scalar other) -> Tensor",
        "aten::ge(Tensor self, Tensor other) -> Tensor",
        "aten::ge(Tensor self, Scalar other) -> Tensor",
        "aten::gt(Tensor self, Tensor other) -> Tensor",
        "aten::gt(Tensor self, Scalar other) -> Tensor",
        "aten::le(Tensor self, Tensor other) -> Tensor",
        "aten::le(Tensor self, Scalar other) -> Tensor",
        "aten::lt(Tensor self, Tensor other) -> Tensor",
        "aten::lt(Tensor self, Scalar other) -> Tensor"};
    for (auto signature : BinaryOp) {
      auto ptr_op = getOperatorForLiteral(signature);
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            static std::unordered_map<Symbol, BinaryOpType> op_mapping(
                {{aten::lt, BinaryOpType::LT},
                 {aten::le, BinaryOpType::LE},
                 {aten::gt, BinaryOpType::GT},
                 {aten::ge, BinaryOpType::GE},
                 {aten::ne, BinaryOpType::NE},
                 {aten::eq, BinaryOpType::Eq}});

            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getPWFormatValues(
                c10::nullopt,
                value_map[node->inputs()[0]->unique()],
                value_map[node->inputs()[1]->unique()]);
            auto lhs = list_val.front();
            list_val.pop_front();
            auto rhs = list_val.front();
            list_val.pop_front();

            auto out = binaryOp(
                op_mapping[node->kind()],
                lhs,
                rhs,
                TypePromotion::comparison_op_config);
            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    std::array<const char*, kNumUnaryOps> UnaryOp = {
        "aten::abs(Tensor self) -> Tensor",
        "aten::bitwise_not(Tensor self) -> Tensor",
        "aten::ceil(Tensor self) -> Tensor",
        "aten::floor(Tensor self) -> Tensor",
        "aten::frac(Tensor self) -> Tensor",
        "aten::neg(Tensor self) -> Tensor",
        "aten::relu(Tensor self) -> Tensor",
        "aten::round(Tensor self) -> Tensor",
        "aten::silu(Tensor self) -> Tensor",
        "aten::trunc(Tensor self) -> Tensor",
    };
    for (auto signature : UnaryOp) {
      auto ptr_op = getOperatorForLiteral(signature);
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            static std::unordered_map<Symbol, UnaryOpType> op_mapping({
                {aten::abs, UnaryOpType::Abs},
                {aten::bitwise_not, UnaryOpType::Not},
                {aten::ceil, UnaryOpType::Ceil},
                {aten::floor, UnaryOpType::Floor},
                {aten::frac, UnaryOpType::Frac},
                {aten::neg, UnaryOpType::Neg},
                {aten::relu, UnaryOpType::Relu},
                {aten::round, UnaryOpType::Round},
                {aten::silu, UnaryOpType::Silu},
                {aten::trunc, UnaryOpType::Trunc},
            });
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt, value_map[node->inputs()[0]->unique()]);
            auto operand = list_val.front();
            list_val.pop_front();
            auto out = unaryOp(op_mapping[node->kind()], operand);
            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    std::array<const char*, kNumUnaryFloatOps> UnaryFloatOp = {
        "aten::log(Tensor self) -> Tensor",
        "aten::log10(Tensor self) -> Tensor",
        "aten::log1p(Tensor self) -> Tensor",
        "aten::log2(Tensor self) -> Tensor",
        "aten::lgamma(Tensor self) -> Tensor",
        "aten::exp(Tensor self) -> Tensor",
        "aten::expm1(Tensor self) -> Tensor",
        "aten::erf(Tensor self) -> Tensor",
        "aten::erfc(Tensor self) -> Tensor",
        "aten::cos(Tensor self) -> Tensor",
        "aten::acos(Tensor self) -> Tensor",
        "aten::cosh(Tensor self) -> Tensor",
        "aten::sin(Tensor self) -> Tensor",
        "aten::asin(Tensor self) -> Tensor",
        "aten::sinh(Tensor self) -> Tensor",
        "aten::tan(Tensor self) -> Tensor",
        "aten::atan(Tensor self) -> Tensor",
        "aten::tanh(Tensor self) -> Tensor",
        "aten::atanh(Tensor self) -> Tensor",
        "aten::sqrt(Tensor self) -> Tensor",
        "aten::rsqrt(Tensor self) -> Tensor",
        "aten::reciprocal(Tensor self) -> Tensor",
        "aten::sigmoid(Tensor self) -> Tensor"};
    for (auto signature : UnaryFloatOp) {
      auto ptr_op = getOperatorForLiteral(signature);
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            static std::unordered_map<Symbol, UnaryOpType> op_mapping({
                {aten::log, UnaryOpType::Log},
                {aten::log10, UnaryOpType::Log10},
                {aten::log1p, UnaryOpType::Log1p},
                {aten::log2, UnaryOpType::Log2},
                {aten::lgamma, UnaryOpType::Lgamma},
                {aten::exp, UnaryOpType::Exp},
                {aten::expm1, UnaryOpType::Expm1},
                {aten::erf, UnaryOpType::Erf},
                {aten::erfc, UnaryOpType::Erfc},
                {aten::cos, UnaryOpType::Cos},
                {aten::acos, UnaryOpType::Acos},
                {aten::cosh, UnaryOpType::Cosh},
                {aten::sin, UnaryOpType::Sin},
                {aten::asin, UnaryOpType::Asin},
                {aten::sinh, UnaryOpType::Sinh},
                {aten::tan, UnaryOpType::Tan},
                {aten::tanh, UnaryOpType::Tanh},
                {aten::atan, UnaryOpType::Atan},
                {aten::atanh, UnaryOpType::Atanh},
                {aten::sqrt, UnaryOpType::Sqrt},
                {aten::rsqrt, UnaryOpType::Rsqrt},
                {aten::reciprocal, UnaryOpType::Reciprocal},
                {aten::sigmoid, UnaryOpType::Sigmoid},
            });
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt, value_map[node->inputs()[0]->unique()]);
            auto operand = list_val.front();
            list_val.pop_front();
            auto out = unaryOp(
                op_mapping[node->kind()],
                operand,
                TypePromotion::float_op_config);
            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    std::array<const char*, kNumUnaryIsOps> UnaryIsOp = {
        "aten::isfinite(Tensor self) -> Tensor",
        "aten::isinf(Tensor self) -> Tensor",
        "aten::isnan(Tensor self) -> Tensor",
        "aten::isneginf(Tensor self) -> Tensor",
        "aten::isposinf(Tensor self) -> Tensor",
        "aten::isreal(Tensor self) -> Tensor"};
    for (auto signature : UnaryIsOp) {
      auto ptr_op = getOperatorForLiteral(signature);
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            static std::unordered_map<Symbol, UnaryOpType> op_mapping({
                {aten::isfinite, UnaryOpType::IsFinite},
                {aten::isinf, UnaryOpType::IsInf},
                {aten::isnan, UnaryOpType::IsNan},
                {aten::isneginf, UnaryOpType::IsNegInf},
                {aten::isposinf, UnaryOpType::IsPosInf},
                {aten::isreal, UnaryOpType::IsReal},
            });
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt, value_map[node->inputs()[0]->unique()]);
            auto operand = list_val.front();
            list_val.pop_front();
            auto out = unaryIsOp(op_mapping[node->kind()], operand);
            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt, value_map[node->inputs()[0]->unique()]);
            auto operand = list_val.front();
            list_val.pop_front();

            if (!node->input(3)->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              auto device = constant_as<c10::Device>(node->input(3));
              TORCH_INTERNAL_ASSERT(
                  device.has_value() && device->is_cuda(),
                  "rand_like in nvfuser is not on cuda device");
              auto input_tensor_type =
                  node->input(0)->type()->cast<TensorType>();
              // device->index() == -1 indicating that we don't change device
              // index
              if (device->index() != -1 && input_tensor_type) {
                auto input_device = input_tensor_type->device();
                // we expect device index to be consistent with input and it
                // should have already been handled by partition
                TORCH_INTERNAL_ASSERT(
                    !input_device.has_value() ||
                        input_device->index() == device->index(),
                    "rand_like in nvfuser is not on cuda device");
              }
            }

            auto out = rand_like(operand);
            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          [](const Node* node) -> bool {
            if (!isInputNonSizeZeroTensor(node)) {
              return false;
            }
            if (!node->input(1)->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get())) ||
                !node->input(2)->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get())) ||
                !node->input(5)->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              return false;
            }
            return true;
          },
          nullptr);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::softplus(Tensor self, Scalar beta, Scalar threshold) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt, value_map[node->inputs()[0]->unique()]);
            auto operand = list_val.front()->as<TensorView>();
            list_val.pop_front();
            auto& beta = value_map[node->inputs()[1]->unique()];
            auto& threshold = value_map[node->inputs()[2]->unique()];
            auto out = softplus(operand, beta, threshold);
            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt, value_map[node->inputs()[0]->unique()]);
            auto operand = list_val.front();
            list_val.pop_front();
            auto& th = value_map[node->inputs()[1]->unique()];
            auto& value = value_map[node->inputs()[2]->unique()];

            auto out = threshold(operand, th, value);
            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    { // LTC uses threshold_backward for relu_backward
      auto ptr_op = getOperatorForLiteral(
          "aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getPWFormatValues(
                c10::nullopt,
                value_map[node->inputs()[0]->unique()],
                value_map[node->inputs()[1]->unique()]);
            auto grad_output = list_val.front();
            list_val.pop_front();
            auto input = list_val.front();
            auto& threshold = value_map[node->inputs()[2]->unique()];

            auto comparison = binaryOp(
                BinaryOpType::GT,
                input,
                threshold,
                TypePromotion::comparison_op_config);
            auto mask = castOp(input->getDataType().value(), comparison);
            auto out = mul(grad_output, mask);

            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::clamp(Tensor self, Scalar? min, Scalar? max) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt, value_map[node->inputs()[0]->unique()]);
            auto operand = list_val.front();
            list_val.pop_front();
            Val* min = value_map.count(node->inputs()[1]->unique()) != 0
                ? *value_map[node->inputs()[1]->unique()]
                : nullptr;
            Val* max = value_map.count(node->inputs()[2]->unique()) != 0
                ? *value_map[node->inputs()[2]->unique()]
                : nullptr;

            Val* out = clamp(operand, min, max);
            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::where(Tensor condition, Tensor self, Tensor other) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getPWFormatValues(
                c10::nullopt,
                value_map[node->inputs()[0]->unique()],
                value_map[node->inputs()[1]->unique()],
                value_map[node->inputs()[2]->unique()]);
            auto condition = list_val.front();
            list_val.pop_front();
            auto x = list_val.front();
            list_val.pop_front();
            auto y = list_val.front();
            list_val.pop_front();

            auto out = where(condition, x, y);
            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    {
      std::array<const char*, kNumLerpOps> LerpOp = {
          "aten::lerp(Tensor self, Tensor end, Scalar weight) -> Tensor",
          "aten::lerp(Tensor self, Tensor end, Tensor weight) -> Tensor"};
      for (auto signature : LerpOp) {
        auto ptr_op = getOperatorForLiteral(signature);
        REGISTER_PARSE_RULE(
            ptr_op,
            {
              MemoryFormat format;
              std::list<Val*> list_val;
              std::tie(format, list_val) = getPWFormatValues(
                  c10::nullopt,
                  value_map[node->inputs()[0]->unique()],
                  value_map[node->inputs()[1]->unique()],
                  value_map[node->inputs()[2]->unique()]);
              auto self = list_val.front();
              list_val.pop_front();
              auto end = list_val.front();
              list_val.pop_front();
              auto weight = list_val.front();
              list_val.pop_front();

              auto out = lerp(self, end, weight);
              value_map.emplace(
                  node->output()->unique(), ValueHolder(out, format));
            },
            isInputNonSizeZeroTensor,
            nullptr);
      }
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getPWFormatValues(
                c10::nullopt,
                value_map[node->inputs()[0]->unique()],
                value_map[node->inputs()[1]->unique()],
                value_map[node->inputs()[2]->unique()],
                value_map[node->inputs()[3]->unique()]);
            auto self = list_val.front();
            list_val.pop_front();
            auto tensor1 = list_val.front();
            list_val.pop_front();
            auto tensor2 = list_val.front();
            list_val.pop_front();
            auto value = list_val.front();
            list_val.pop_front();

            auto out = addcmul(self, tensor1, tensor2, value);
            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt,
                value_map[node->inputs()[0]->unique()],
                value_map[node->inputs()[1]->unique()]);
            auto input = list_val.front();
            list_val.pop_front();
            auto prob = list_val.front();
            list_val.pop_front();
            auto train = constant_as<bool>(node->input(2));

            TORCH_INTERNAL_ASSERT(
                train.has_value(), "dropout needs constant `train` flag");

            if (train.value()) {
              auto result = dropout(input->as<TensorView>(), prob);

              value_map.emplace(
                  node->output(0)->unique(),
                  ValueHolder(result.output, format));
              value_map.emplace(
                  node->output(1)->unique(), ValueHolder(result.mask, format));
            } else {
              value_map.emplace(node->output(0)->unique(), input);
              value_map.emplace(
                  node->output(1)->unique(),
                  ValueHolder(TensorViewBuilder().build(), format));
            }
          },
          [](const Node* node) -> bool {
            if (!isInputNonSizeZeroTensor(node)) {
              return false;
            }
            if (node->inputs()[2]->node()->kind() != prim::Constant) {
              return false;
            }
            return true;
          },
          nullptr);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::dropout(Tensor input, float p, bool train) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt,
                value_map[node->inputs()[0]->unique()],
                value_map[node->inputs()[1]->unique()]);
            auto input = list_val.front();
            list_val.pop_front();
            auto prob = list_val.front();
            list_val.pop_front();

            auto train = constant_as<bool>(node->input(2));
            TORCH_INTERNAL_ASSERT(
                train.has_value(), "dropout needs constant `train` flag");

            if (train.value()) {
              auto result = dropout(input->as<TensorView>(), prob);

              value_map.emplace(
                  node->output()->unique(), ValueHolder(result.output, format));
            } else {
              value_map.emplace(
                  node->output()->unique(), ValueHolder(input, format));
            }
          },
          [](const Node* node) -> bool {
            if (!isInputNonSizeZeroTensor(node)) {
              return false;
            }
            if (node->inputs()[2]->node()->kind() != prim::Constant) {
              return false;
            }
            return true;
          },
          nullptr);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::native_dropout_backward(Tensor grad_output, Tensor mask, float scale) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getPWFormatValues(
                c10::nullopt,
                value_map[node->inputs()[0]->unique()],
                value_map[node->inputs()[1]->unique()],
                value_map[node->inputs()[2]->unique()]);
            auto grad = list_val.front();
            list_val.pop_front();
            auto mask = list_val.front();
            list_val.pop_front();
            auto scale = list_val.front();
            list_val.pop_front();

            auto output = dropout_backward(
                grad->as<TensorView>(), mask->as<TensorView>(), scale);
            value_map.emplace(
                node->output()->unique(), ValueHolder(output, format));
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    {
      std::array<const char*, kNumInstancenormFwd> InstanceNormFwd = {
          "aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor"};
      for (auto signature : InstanceNormFwd) {
        auto ptr_op = getOperatorForLiteral(signature);
        REGISTER_PARSE_RULE(
            ptr_op,
            {
              // TODO: handle channels last
              MemoryFormat format;
              std::list<Val*> list_val;
              std::tie(format, list_val) = getConsistentValues(
                  MemoryFormat::Contiguous(),
                  value_map[node->inputs()[0]->unique()]);
              auto input_t = list_val.front();
              list_val.pop_front();
              auto input = input_t->as<TensorView>();

              TensorView* weight = nullptr;
              if (!node->input(1)->type()->isSubtypeOf(
                      static_cast<c10::TypePtr>(NoneType::get()))) {
                weight = value_map[node->input(1)->unique()]->as<TensorView>();
              }

              TensorView* bias = nullptr;
              if (!node->input(2)->type()->isSubtypeOf(
                      static_cast<c10::TypePtr>(NoneType::get()))) {
                bias = value_map[node->input(2)->unique()]->as<TensorView>();
              }

              TensorView* running_mean = nullptr;
              if (!node->input(3)->type()->isSubtypeOf(
                      static_cast<c10::TypePtr>(NoneType::get()))) {
                running_mean =
                    value_map[node->input(3)->unique()]->as<TensorView>();
              }

              TensorView* running_var = nullptr;
              if (!node->input(4)->type()->isSubtypeOf(
                      static_cast<c10::TypePtr>(NoneType::get()))) {
                running_var =
                    value_map[node->input(4)->unique()]->as<TensorView>();
              }

              // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
              auto use_input_stats = constant_as<bool>(node->input(5));
              TORCH_INTERNAL_ASSERT(
                  use_input_stats.has_value(),
                  "The use_input_stats (bool) parameter is required.");
              const bool kUseInputStats = use_input_stats.value();

              Val* momentum_ptr = nullptr;
              // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
              if (auto momentum = constant_as<float>(node->input(6))) {
                momentum_ptr = IrBuilder::create<Double>(momentum.value());
              } else {
                // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
                momentum_ptr = value_map[node->input(6)->unique()];
              }

              Val* eps_ptr = nullptr;
              // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
              if (auto eps = constant_as<float>(node->input(7))) {
                eps_ptr = IrBuilder::create<Double>(eps.value());
              } else {
                // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
                eps_ptr = value_map[node->input(7)->unique()];
              }

              auto result = instance_norm(
                  input,
                  weight,
                  bias,
                  running_mean,
                  running_var,
                  kUseInputStats,
                  momentum_ptr,
                  eps_ptr);

              if (node->kind() ==
                  c10::Symbol::fromQualString("aten::instance_norm")) {
                value_map.emplace(node->output()->unique(), result.output);
              }
            },
            [](const Node* node) -> bool {
              if (isReductionNonCompatibleTensor(
                      node->input(0)->type()->cast<TensorType>())) {
                return false;
              }
              return true;
            },
            [](const Node* node) -> OperatorType {
              return OperatorType::Normalization;
            });
      }
    }

    {
      std::array<const char*, kNumBatchnormFwd> BatchNormFwd = {
          "aten::_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, Tensor, int)",
          "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)",
          "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor"};
      for (auto signature : BatchNormFwd) {
        auto ptr_op = getOperatorForLiteral(signature);
        REGISTER_PARSE_RULE(
            ptr_op,
            {
              MemoryFormat format;
              Val* operand = nullptr;
              std::tie(format, operand) =
                  value_map[node->input(0)->unique()].getEntry();
              if (format.hasPermutation() && !format.isChannelsLast()) {
                format = MemoryFormat::Contiguous();
                operand = value_map[node->input(0)->unique()].maybeConvertValue(
                    format);
              }
              auto input = operand->as<TensorView>();

              TensorView* weight = nullptr;
              if (!node->input(1)->type()->isSubtypeOf(
                      static_cast<c10::TypePtr>(NoneType::get()))) {
                weight = value_map[node->input(1)->unique()]->as<TensorView>();
              }

              TensorView* bias = nullptr;
              if (!node->input(2)->type()->isSubtypeOf(
                      static_cast<c10::TypePtr>(NoneType::get()))) {
                bias = value_map[node->input(2)->unique()]->as<TensorView>();
              }

              // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
              auto training = constant_as<bool>(node->input(5));
              TORCH_INTERNAL_ASSERT(
                  training.has_value(),
                  "The training (bool) parameter is required.");
              const bool kTraining = training.value();

              TensorView* running_mean = nullptr;
              if (!node->input(3)->type()->isSubtypeOf(
                      static_cast<c10::TypePtr>(NoneType::get()))) {
                running_mean =
                    value_map[node->input(3)->unique()]->as<TensorView>();
              }

              TensorView* running_var = nullptr;
              if (!node->input(4)->type()->isSubtypeOf(
                      static_cast<c10::TypePtr>(NoneType::get()))) {
                running_var =
                    value_map[node->input(4)->unique()]->as<TensorView>();
              }

              Val* momentum_ptr = nullptr;
              // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
              if (auto momentum = constant_as<float>(node->input(6))) {
                momentum_ptr = IrBuilder::create<Double>(momentum.value());
              } else {
                // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
                momentum_ptr = value_map[node->input(6)->unique()];
              }

              Val* eps_ptr = nullptr;
              // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
              if (auto eps = constant_as<float>(node->input(7))) {
                eps_ptr = IrBuilder::create<Double>(eps.value());
              } else {
                // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
                eps_ptr = value_map[node->input(7)->unique()];
              }

              auto result = batch_norm(
                  input,
                  weight,
                  bias,
                  running_mean,
                  running_var,
                  kTraining,
                  momentum_ptr,
                  eps_ptr,
                  format.isChannelsLast());

              if (node->kind() ==
                      c10::Symbol::fromQualString("aten::native_batch_norm") ||
                  node->kind() ==
                      c10::Symbol::fromQualString(
                          "aten::_batch_norm_impl_index")) {
                // TODO: output 3 & 4 are not created
                //       we are not creating these outputs because codegen
                //       currently lacks the support.
                value_map.emplace(
                    node->output(0)->unique(),
                    ValueHolder(result.output, format));
                value_map.emplace(node->output(1)->unique(), result.mean);
                value_map.emplace(node->output(2)->unique(), result.invstd);
              } else if (
                  node->kind() ==
                  c10::Symbol::fromQualString("aten::batch_norm")) {
                value_map.emplace(
                    node->output()->unique(),
                    ValueHolder(result.output, format));
              }
            },
            [](const Node* node) -> bool {
              if (isReductionNonCompatibleTensor(
                      node->input(0)->type()->cast<TensorType>())) {
                return false;
              }
              if (node->input(5)->node()->kind() != prim::Constant) {
                return false;
              }
              return true;
            },
            [](const Node* node) -> OperatorType {
              return OperatorType::Normalization;
            });
      }
    }

    {
      std::array<const char*, kNumBatchnormBwd> BatchNormBwd = {
          "aten::_batch_norm_impl_index_backward(int impl_index, Tensor input, Tensor grad_output, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var_transform, bool train, float eps, bool[3] output_mask, Tensor reservedSpace) -> (Tensor, Tensor, Tensor)",
          "aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"};
      for (auto signature : BatchNormBwd) {
        auto ptr_op = getOperatorForLiteral(signature);
        REGISTER_PARSE_RULE(
            ptr_op,
            {
              JitValue* ts_input = nullptr;
              JitValue* ts_grad_output;
              JitValue* ts_weight = nullptr;
              JitValue* ts_r_mean = nullptr;
              JitValue* ts_r_var = nullptr;
              JitValue* ts_save_mean = nullptr;
              JitValue* ts_save_invstd = nullptr;
              JitValue* ts_train = nullptr;
              JitValue* ts_eps = nullptr;
              JitValue* ts_mask = nullptr;
              if (node->kind() ==
                  c10::Symbol::fromQualString(
                      "aten::_batch_norm_impl_index_backward")) {
                ts_input = node->input(1);
                ts_grad_output = node->input(2);
                ts_weight = node->input(3);
                ts_r_mean = node->input(4);
                ts_r_var = node->input(5);
                ts_save_mean = node->input(6);
                ts_save_invstd = node->input(7);
                ts_train = node->input(8);
                ts_eps = node->input(9);
                ts_mask = node->input(10);
              } else if (
                  node->kind() ==
                  c10::Symbol::fromQualString(
                      "aten::native_batch_norm_backward")) {
                ts_grad_output = node->input(0);
                ts_input = node->input(1);
                ts_weight = node->input(2);
                ts_r_mean = node->input(3);
                ts_r_var = node->input(4);
                ts_save_mean = node->input(5);
                ts_save_invstd = node->input(6);
                ts_train = node->input(7);
                ts_eps = node->input(8);
                ts_mask = node->input(9);
              } else {
                TORCH_INTERNAL_ASSERT(
                    false,
                    "Forgot to register the key for BN variation: ",
                    node->kind().toDisplayString());
              }

              // discard impl_index and reservedSpace since we don't use them
              MemoryFormat format;
              std::list<Val*> list_val;
              std::tie(format, list_val) = getConsistentValues(
                  c10::nullopt,
                  value_map[ts_input->unique()],
                  value_map[ts_grad_output->unique()]);
              if (format.hasPermutation() && !format.isChannelsLast()) {
                std::tie(format, list_val) = getConsistentValues(
                    MemoryFormat::Contiguous(),
                    value_map[ts_input->unique()],
                    value_map[ts_grad_output->unique()]);
              }
              auto operand0 = list_val.front();
              list_val.pop_front();
              auto operand1 = list_val.front();
              list_val.pop_front();
              auto input = operand0->as<TensorView>();
              auto grad_out = operand1->as<TensorView>();

              TensorView* weight = nullptr;
              if (!ts_weight->type()->isSubtypeOf(
                      static_cast<c10::TypePtr>(NoneType::get()))) {
                weight = value_map[ts_weight->unique()]->as<TensorView>();
              }

              TensorView* running_mean = nullptr;
              if (!ts_r_mean->type()->isSubtypeOf(
                      static_cast<c10::TypePtr>(NoneType::get()))) {
                running_mean = value_map[ts_r_mean->unique()]->as<TensorView>();
              }

              TensorView* running_var = nullptr;
              if (!ts_r_var->type()->isSubtypeOf(
                      static_cast<c10::TypePtr>(NoneType::get()))) {
                running_var = value_map[ts_r_var->unique()]->as<TensorView>();
              }

              TensorView* save_mean = nullptr;
              // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
              if (!ts_save_mean->type()->isSubtypeOf(
                      static_cast<c10::TypePtr>(NoneType::get()))) {
                // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
                save_mean = value_map[ts_save_mean->unique()]->as<TensorView>();
              }

              TensorView* save_invstd = nullptr;
              // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
              if (!ts_save_invstd->type()->isSubtypeOf(
                      static_cast<c10::TypePtr>(NoneType::get()))) {
                save_invstd =
                    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
                    value_map[ts_save_invstd->unique()]->as<TensorView>();
              }

              // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
              auto training = constant_as<bool>(ts_train);
              TORCH_INTERNAL_ASSERT(
                  training.has_value(),
                  "The training (bool) parameter is required.");
              const bool kTraining = training.value();

              // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
              Val* eps_ptr = nullptr;
              // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
              if (auto eps = constant_as<float>(ts_eps)) {
                eps_ptr = IrBuilder::create<Double>(eps.value());
              } else {
                // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
                eps_ptr = value_map[ts_eps->unique()];
              }

              // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
              auto out_mask_list = constant_as<c10::List<bool>>(ts_mask);
              TORCH_INTERNAL_ASSERT(
                  out_mask_list.has_value(),
                  "output mask for batch_norm_backward");
              std::vector<bool> output_mask;
              for (const auto value : out_mask_list->vec()) {
                output_mask.emplace_back(static_cast<bool>(value));
              }

              // TODO: merge this loop below.
              if (kTraining) {
                TORCH_INTERNAL_ASSERT(
                    save_mean != nullptr && save_invstd != nullptr,
                    "When training=True, save_mean and save_invstd are required.");
              } else {
                // TODO: this is not a legit assumption? Can't we run with
                // track_running_stats == false && training == false
                // which should just run through the case above.
                TORCH_INTERNAL_ASSERT(
                    running_mean != nullptr && running_var != nullptr,
                    "When training=False, running_mean and running_invstd are required.");
              }

              auto grads = batch_norm_backward(
                  input,
                  grad_out,
                  weight,
                  running_mean,
                  running_var,
                  save_mean,
                  save_invstd,
                  kTraining,
                  eps_ptr,
                  output_mask,
                  format.isChannelsLast());

              if (output_mask[0]) {
                TORCH_INTERNAL_ASSERT(grads.grad_input != nullptr);
                value_map.emplace(
                    node->output(0)->unique(),
                    ValueHolder(grads.grad_input, format));
              } else {
                TORCH_INTERNAL_ASSERT(grads.grad_input == nullptr);
                value_map.emplace(
                    node->output(0)->unique(),
                    ValueHolder(TensorViewBuilder().build(), format));
              }

              if (output_mask[1]) {
                TORCH_INTERNAL_ASSERT(grads.grad_weight != nullptr);
                value_map.emplace(node->output(1)->unique(), grads.grad_weight);
              } else {
                TORCH_INTERNAL_ASSERT(grads.grad_weight == nullptr);
                value_map.emplace(
                    node->output(1)->unique(), TensorViewBuilder().build());
              }

              if (output_mask[2]) {
                TORCH_INTERNAL_ASSERT(grads.grad_bias != nullptr);
                value_map.emplace(node->output(2)->unique(), grads.grad_bias);
              } else {
                TORCH_INTERNAL_ASSERT(grads.grad_bias == nullptr);
                value_map.emplace(
                    node->output(2)->unique(), TensorViewBuilder().build());
              }
            },
            [](const Node* node) -> bool {
              if (isReductionNonCompatibleTensor(
                      node->input(1)->type()->cast<TensorType>())) {
                return false;
              }
              if (node->kind() ==
                  c10::Symbol::fromQualString(
                      "aten::_batch_norm_impl_index_backward")) {
                if (node->inputs()[8]->node()->kind() != prim::Constant) {
                  return false;
                }
                if (node->inputs()[10]->node()->kind() != prim::Constant) {
                  return false;
                }
              } else if (
                  node->kind() ==
                  c10::Symbol::fromQualString(
                      "aten::native_batch_norm_backward")) {
                if (node->inputs()[7]->node()->kind() != prim::Constant) {
                  return false;
                }
                if (node->inputs()[9]->node()->kind() != prim::Constant) {
                  return false;
                }
              } else {
                TORCH_INTERNAL_ASSERT(
                    false,
                    "Forgot to update profiled constant check for",
                    node->kind().toDisplayString());
              }
              return true;
            },
            [](const Node* node) -> OperatorType {
              return OperatorType::Normalization;
            });
      }
    }

    {
      std::array<const char*, kNumLayernormFwd> LayerNormFwd = {
          "aten::native_layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)",
          "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor"};
      for (auto signature : LayerNormFwd) {
        auto ptr_op = getOperatorForLiteral(signature);
        REGISTER_PARSE_RULE(
            ptr_op,
            {
              MemoryFormat format;
              std::list<Val*> list_val;
              std::tie(format, list_val) = getConsistentValues(
                  MemoryFormat::Contiguous(),
                  value_map[node->inputs()[0]->unique()]);
              auto input_t = list_val.front();
              list_val.pop_front();
              auto input = input_t->as<TensorView>();

              auto norm_shape_optional =
                  constant_as<c10::List<int64_t>>(node->input(1));
              TORCH_INTERNAL_ASSERT(
                  norm_shape_optional.has_value(),
                  "The Normalized_Shape list is required.");
              auto norm_shape = norm_shape_optional->vec();

              TensorView* weight = nullptr;
              if (!node->input(2)->type()->isSubtypeOf(
                      static_cast<c10::TypePtr>(NoneType::get()))) {
                weight = value_map[node->input(2)->unique()]->as<TensorView>();
              }

              TensorView* bias = nullptr;
              if (!node->input(3)->type()->isSubtypeOf(
                      static_cast<c10::TypePtr>(NoneType::get()))) {
                bias = value_map[node->input(3)->unique()]->as<TensorView>();
              }

              Val* eps_ptr = nullptr;
              if (auto eps = constant_as<float>(node->input(4))) {
                eps_ptr = IrBuilder::create<Double>(eps.value());
              } else {
                eps_ptr = value_map[node->input(4)->unique()];
              }

              auto result =
                  layer_norm(input, norm_shape, weight, bias, eps_ptr);

              if (node->kind() ==
                  c10::Symbol::fromQualString("aten::native_layer_norm")) {
                value_map.emplace(node->output(0)->unique(), result.output);
                value_map.emplace(node->output(1)->unique(), result.mean);
                value_map.emplace(node->output(2)->unique(), result.invstd);
              } else if (
                  node->kind() ==
                  c10::Symbol::fromQualString("aten::layer_norm")) {
                value_map.emplace(node->output()->unique(), result.output);
              }
            },
            // TODO: #ProfileIValue List should update this
            [](const Node* node) -> bool {
              if (isReductionNonCompatibleTensor(
                      node->input(0)->type()->cast<TensorType>())) {
                return false;
              }
              if (node->inputs()[1]->node()->kind() != prim::Constant) {
                return false;
              }
              return true;
            },
            [](const Node* node) -> OperatorType {
              return OperatorType::Normalization;
            });
      }
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::native_layer_norm_backward(Tensor grad_out, Tensor input, int[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                MemoryFormat::Contiguous(),
                value_map[node->inputs()[0]->unique()],
                value_map[node->inputs()[1]->unique()]);
            auto grad_out_t = list_val.front();
            list_val.pop_front();
            auto input_t = list_val.front();
            list_val.pop_front();
            auto grad_out = grad_out_t->as<TensorView>();
            auto input = input_t->as<TensorView>();

            auto norm_shape_optional =
                constant_as<c10::List<int64_t>>(node->input(2));
            TORCH_INTERNAL_ASSERT(
                norm_shape_optional.has_value(),
                "The Normalized_Shape list is required.");
            auto norm_shape = norm_shape_optional->vec();

            auto mean = value_map[node->input(3)->unique()]->as<TensorView>();
            auto rstd = value_map[node->input(4)->unique()]->as<TensorView>();

            TensorView* weight = nullptr;
            // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
            if (!node->input(5)->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
              weight = value_map[node->input(5)->unique()]->as<TensorView>();
            }

            TensorView* bias = nullptr;
            // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
            if (!node->input(6)->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
              bias = value_map[node->input(6)->unique()]->as<TensorView>();
            }

            // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
            auto output_mask_optional =
                constant_as<c10::List<bool>>(node->input(7));
            TORCH_INTERNAL_ASSERT(
                output_mask_optional.has_value(),
                "output mask for layer_norm_backward");
            std::vector<bool> output_mask = output_mask_optional->vec();

            auto grad = layer_norm_backward(
                grad_out,
                input,
                norm_shape,
                mean,
                rstd,
                weight,
                bias,
                output_mask);

            if (output_mask[0]) {
              TORCH_INTERNAL_ASSERT(grad.grad_input != nullptr);
              value_map.emplace(node->output(0)->unique(), grad.grad_input);
            } else {
              TORCH_INTERNAL_ASSERT(grad.grad_input == nullptr);
              value_map.emplace(
                  node->output(0)->unique(), TensorViewBuilder().build());
            }

            if (output_mask[1] && weight != nullptr) {
              TORCH_INTERNAL_ASSERT(grad.grad_weight != nullptr);
              value_map.emplace(node->output(1)->unique(), grad.grad_weight);
            } else {
              TORCH_INTERNAL_ASSERT(grad.grad_weight == nullptr);
              value_map.emplace(
                  node->output(1)->unique(), TensorViewBuilder().build());
            }

            if (output_mask[2] && bias != nullptr) {
              TORCH_INTERNAL_ASSERT(grad.grad_bias != nullptr);
              value_map.emplace(node->output(2)->unique(), grad.grad_bias);
            } else {
              TORCH_INTERNAL_ASSERT(grad.grad_bias == nullptr);
              value_map.emplace(
                  node->output(2)->unique(), TensorViewBuilder().build());
            }
          },
          // TODO: #ProfileIValue List should update this
          [](const Node* node) -> bool {
            if (isReductionNonCompatibleTensor(
                    node->input(0)->type()->cast<TensorType>())) {
              return false;
            }
            if (node->inputs()[2]->node()->kind() != prim::Constant) {
              return false;
            }
            if (node->inputs()[7]->node()->kind() != prim::Constant) {
              return false;
            }
            return true;
          },
          [](const Node* node) -> OperatorType {
            return OperatorType::Normalization;
          });
    }

    {
      std::array<const char*, kNumSoftmaxFwd> SoftmaxFwd = {
          "aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
          "aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor"};
      for (auto signature : SoftmaxFwd) {
        auto ptr_op = getOperatorForLiteral(signature);
        REGISTER_PARSE_RULE(
            ptr_op,
            {
              MemoryFormat format;
              std::list<Val*> list_val;
              std::tie(format, list_val) = getConsistentValues(
                  MemoryFormat::Contiguous(),
                  value_map[node->inputs()[0]->unique()]);
              auto input_t = list_val.front();
              list_val.pop_front();
              auto input = input_t->as<TensorView>();

              auto dim_value = constant_as<int>(node->input(1));
              TORCH_INTERNAL_ASSERT(
                  dim_value.has_value(), "dim in softmax is not valid");

              auto data_type = DataType::Null;
              if (const auto opt_ivalue = toIValue(node->input(2))) {
                if (!opt_ivalue->isNone()) {
                  data_type = aten_to_data_type(opt_ivalue->toScalarType());
                }
              }

              input = (data_type != DataType::Null)
                  ? optionalCastStrict(data_type, input)->as<TensorView>()
                  : input;

              bool is_log_softmax = node->kind() ==
                  c10::Symbol::fromQualString("aten::log_softmax");

              auto output = (is_log_softmax)
                  ? log_softmax(input, dim_value.value())
                  : softmax(input, dim_value.value());

              value_map.emplace(node->output()->unique(), output);
            },
            [](const Node* node) -> bool {
              if (isReductionNonCompatibleTensor(
                      node->input(0)->type()->cast<TensorType>())) {
                return false;
              }
              if (node->inputs()[1]->node()->kind() != prim::Constant) {
                return false;
              }
              if (!isScalarTypeCompatible(node, 2)) {
                return false;
              }
              return true;
            },
            [](const Node* node) -> OperatorType {
              return OperatorType::Normalization;
            });
      }
    }

    { // LTC uses this op for softmax
      auto ptr_op = getOperatorForLiteral(
          "aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                MemoryFormat::Contiguous(),
                value_map[node->inputs()[0]->unique()]);
            auto input_t = list_val.front();
            list_val.pop_front();
            auto input = input_t->as<TensorView>();

            auto dim_value = constant_as<int>(node->input(1));
            TORCH_INTERNAL_ASSERT(
                dim_value.has_value(), "dim in softmax is not valid");

            auto output = softmax(input, dim_value.value());
            value_map.emplace(node->output()->unique(), output);
          },
          [](const Node* node) -> bool {
            if (isReductionNonCompatibleTensor(
                    node->input(0)->type()->cast<TensorType>())) {
              return false;
            }
            if (node->inputs()[1]->node()->kind() != prim::Constant) {
              return false;
            }
            if (node->inputs()[2]->node()->kind() != prim::Constant) {
              return false;
            } else {
              const auto half_to_float = constant_as<bool>(node->input(2));
              TORCH_INTERNAL_ASSERT(
                  half_to_float.has_value(), "Bool half_to_float is not valid");
              auto input_tensor_type =
                  node->input(0)->type()->cast<TensorType>();
              if (half_to_float.value() &&
                  input_tensor_type->scalarType() != at::ScalarType::Half) {
                return false;
              }
            }
            return true;
          },
          [](const Node* node) -> OperatorType {
            return OperatorType::Normalization;
          });
    }

    {
      std::array<const char*, kNumSoftmaxBwd> SoftmaxBwd = {
          "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor",
          "aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor"};
      for (auto signature : SoftmaxBwd) {
        auto ptr_op = getOperatorForLiteral(signature);
        REGISTER_PARSE_RULE(
            ptr_op,
            {
              MemoryFormat format;
              std::list<Val*> list_val;
              std::tie(format, list_val) = getConsistentValues(
                  MemoryFormat::Contiguous(),
                  value_map[node->inputs()[0]->unique()],
                  value_map[node->inputs()[1]->unique()]);
              auto grad_output_t = list_val.front();
              list_val.pop_front();
              auto grad_output = grad_output_t->as<TensorView>();

              auto output_t = list_val.front();
              list_val.pop_front();
              auto output = output_t->as<TensorView>();

              auto dim_value = constant_as<int>(node->input(2));
              TORCH_INTERNAL_ASSERT(
                  dim_value.has_value(), "dim in softmax is not valid");

              // input_dtype here is ignored! type_inference handles it
              bool is_log_softmax = node->kind() ==
                  c10::Symbol::fromQualString(
                                        "aten::_log_softmax_backward_data");
              auto grad_input = (is_log_softmax)
                  ? log_softmax_backward(grad_output, output, dim_value.value())
                  : softmax_backward(grad_output, output, dim_value.value());

              value_map.emplace(node->output()->unique(), grad_input);
            },
            [](const Node* node) -> bool {
              if (isReductionNonCompatibleTensor(
                      node->input(0)->type()->cast<TensorType>())) {
                return false;
              }
              if (node->inputs()[2]->node()->kind() != prim::Constant) {
                return false;
              }
              if (node->inputs()[3]->node()->kind() != prim::Constant) {
                return false;
              }
              return true;
            },
            [](const Node* node) -> OperatorType {
              return OperatorType::Normalization;
            });
      }
    }

    {
      std::array<const char*, kNumVarOps> Variance = {
          "aten::var.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor",
          "aten::std.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor"};
      for (auto signature : Variance) {
        auto ptr_op = getOperatorForLiteral(signature);
        REGISTER_PARSE_RULE(
            ptr_op,
            {
              MemoryFormat format;
              std::list<Val*> list_val;
              std::tie(format, list_val) = getConsistentValues(
                  MemoryFormat::Contiguous(),
                  value_map[node->inputs()[0]->unique()]);
              auto input_t = list_val.front();
              list_val.pop_front();
              auto input = input_t->as<TensorView>();

              bool is_variance =
                  node->kind() == c10::Symbol::fromQualString("aten::var");

              auto dims_list = constant_as<c10::List<int64_t>>(node->input(1));
              TORCH_INTERNAL_ASSERT(
                  dims_list.has_value(), "Cannot fuse with dynamic axes");
              std::vector<int> dims;
              if (!dims_list->empty()) {
                for (const auto dim : dims_list->vec()) {
                  dims.emplace_back(static_cast<int>(dim));
                }
              } else {
                dims.resize(input->as<TensorView>()->nDims());
                std::iota(dims.begin(), dims.end(), 0);
              }

              auto unbiased = constant_as<bool>(node->input(2));
              TORCH_INTERNAL_ASSERT(
                  unbiased.has_value(), "Cannot fuse with dynamic unbiased");

              auto keepdim = constant_as<bool>(node->input(3));
              TORCH_INTERNAL_ASSERT(
                  keepdim.has_value(), "Cannot fuse with dynamic keepdim");

              auto output = (is_variance)
                  ? variance(input, dims, unbiased.value(), keepdim.value())
                  : standard_deviation(
                        input, dims, unbiased.value(), keepdim.value());
              value_map.emplace(node->output()->unique(), output);
            },
            [](const Node* node) -> bool {
              if (isReductionNonCompatibleTensor(
                      node->input(0)->type()->cast<TensorType>())) {
                return false;
              }
              return true;
            },
            [](const Node* node) -> OperatorType {
              return OperatorType::Normalization;
            });
      }
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            // TODO: support channels last in sum
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                MemoryFormat::Contiguous(),
                value_map[node->inputs()[0]->unique()]);
            auto self = list_val.front();
            list_val.pop_front();
            auto dims_list = constant_as<c10::List<int64_t>>(node->input(1));
            TORCH_INTERNAL_ASSERT(
                dims_list.has_value(),
                "aten::sum cannot be fused with dynamic axes");
            std::vector<int> dims;
            if (!dims_list->empty()) {
              for (const auto dim : dims_list->vec()) {
                dims.emplace_back(static_cast<int>(dim));
              }
            } else {
              dims.resize(self->as<TensorView>()->nDims());
              std::iota(dims.begin(), dims.end(), 0);
            }
            auto keepdim = constant_as<bool>(node->input(2));
            TORCH_INTERNAL_ASSERT(
                keepdim.has_value(),
                "aten::sum cannot be fused with dynamic keepdim");
            auto out = sum(self->as<TensorView>(), dims, keepdim.value());
            value_map.emplace(node->output()->unique(), out);
          },
          [](const Node* node) -> bool {
            if (isReductionNonCompatibleTensor(
                    node->input(0)->type()->cast<TensorType>())) {
              return false;
            }
            // TODO: support cast of output types
            if (!node->inputs()[3]->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              // We can only handle output as half, float, and double;
              if (const auto opt_ivalue = toIValue(node->input(3))) {
                const auto scalar_type = opt_ivalue->toScalarType();
                if (!at::isFloatingType(scalar_type)) {
                  return false;
                }
              }
            }
            // we don't support dynamic reduction axes;
            if (node->inputs()[1]->node()->kind() != prim::Constant) {
              return false;
            }
            // we don't support dynamic keepdim yet;
            if (node->inputs()[2]->node()->kind() != prim::Constant) {
              return false;
            }
            return true;
          },
          [](const Node* node) -> OperatorType {
            return OperatorType::Reduction;
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                MemoryFormat::Contiguous(),
                value_map[node->inputs()[0]->unique()]);
            auto operand = list_val.front();
            list_val.pop_front();
            auto self = operand->as<TensorView>();
            auto dims_list = constant_as<c10::List<int64_t>>(node->input(1));
            TORCH_INTERNAL_ASSERT(
                dims_list.has_value(),
                "aten::mean cannot be fused with dynamic axes");
            std::vector<int> dims;
            if (!dims_list->empty()) {
              for (const auto dim : dims_list->vec()) {
                dims.emplace_back(static_cast<int>(dim));
              }
            } else {
              dims.resize(self->as<TensorView>()->nDims());
              std::iota(dims.begin(), dims.end(), 0);
            }
            auto keepdim = constant_as<bool>(node->input(2));
            TORCH_INTERNAL_ASSERT(
                keepdim.has_value(),
                "aten::mean cannot be fused with dynamic keepdim");
            auto o_sum = sum(self, dims, keepdim.value());
            Val* num_features = IrBuilder::create<Double>(1);
            for (auto axis : dims) {
              if (axis < 0) {
                axis += int(self->nDims());
              }
              num_features =
                  mul(num_features, self->domain()->domain()[axis]->extent());
            }
            auto out = div(o_sum, num_features);
            value_map.emplace(node->output()->unique(), out);
          },
          [](const Node* node) -> bool {
            if (isReductionNonCompatibleTensor(
                    node->input(0)->type()->cast<TensorType>())) {
              return false;
            }
            // TODO: support cast of output types
            if (!node->inputs()[3]->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              // We can only handle output as half, float, and double;
              if (const auto opt_ivalue = toIValue(node->input(3))) {
                const auto scalar_type = opt_ivalue->toScalarType();
                if (!at::isFloatingType(scalar_type)) {
                  return false;
                }
              }
            }
            // we don't support dynamic reduction axes;
            if (node->inputs()[1]->node()->kind() != prim::Constant) {
              return false;
            }
            // we don't support dynamic keepdim yet;
            if (node->inputs()[2]->node()->kind() != prim::Constant) {
              return false;
            }
            return true;
          },
          [](const Node* node) -> OperatorType {
            return OperatorType::Reduction;
          });
    }
    {
      std::array<const char*, kNumSumToSize> SumToSize = {
          "aten::_grad_sum_to_size(Tensor(a) self, int[]? size) -> Tensor(a)",
          "aten::sum_to_size(Tensor self, int[] size) -> Tensor"};
      for (auto signature : SumToSize) {
        auto ptr_op = getOperatorForLiteral(signature);
        REGISTER_PARSE_RULE(
            ptr_op,
            {
              MemoryFormat format;
              std::list<Val*> list_val;
              std::tie(format, list_val) = getConsistentValues(
                  MemoryFormat::Contiguous(),
                  value_map[node->inputs()[0]->unique()]);
              auto self = list_val.front();
              list_val.pop_front();
              auto size_to = constant_as<c10::List<int64_t>>(node->input(1));
              TORCH_INTERNAL_ASSERT(
                  size_to.has_value(),
                  "aten::sum cannot be fused with dynamic axes");
              if (!size_to->empty()) {
                auto input = self->as<TensorView>();
                auto out = sum_to(input, size_to->vec());
                // this copy is not necessary, but making copy avoids tricky
                // computational graph where no-op could be challenging.
                if (out == input) {
                  out = set(input);
                }
                value_map.emplace(node->output()->unique(), out);
              } else {
                // We are introducing alias here!
                value_map.emplace(node->output()->unique(), self);
              }
            },
            [](const Node* node) -> bool {
              if (isReductionNonCompatibleTensor(
                      node->input(0)->type()->cast<TensorType>())) {
                return false;
              }
              // we don't support dynamic reduction axes;
              if (node->inputs()[1]->node()->kind() != prim::Constant) {
                return false;
              }
              return true;
            },
            [](const Node* node) -> OperatorType {
              auto size_to = constant_as<c10::List<int64_t>>(node->input(1));
              // technically size_to->empty() should never occur, as specialized
              // _grad_sum_to_size should have been removed by optimization pass
              if (size_to->empty()) {
                return OperatorType::ElementWise;
              } else {
                return OperatorType::ReductionToSize;
              }
            });
      }
    }

    {
      std::array<const char*, kNumAutocastOps> AutocastOps = {
          "aten::_autocast_to_reduced_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled, ScalarType cuda_dtype, ScalarType cpu_dtype) -> Tensor(a)",
          "aten::_autocast_to_full_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled) -> Tensor(a)"};
      for (auto signature : AutocastOps) {
        auto ptr_op = getOperatorForLiteral(signature);
        REGISTER_PARSE_RULE(
            ptr_op,
            {
              MemoryFormat format;
              std::list<Val*> list_val;
              std::tie(format, list_val) = getConsistentValues(
                  c10::nullopt, value_map[node->inputs()[0]->unique()]);
              auto self = list_val.front();
              list_val.pop_front();

              auto out = set(self);
              value_map.emplace(
                  node->output()->unique(), ValueHolder(out, format));
            },
            isInputNonSizeZeroTensor,
            nullptr);
      }
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt, value_map[node->inputs()[0]->unique()]);
            auto self = list_val.front();
            list_val.pop_front();

            auto out = castTensoToDtype(self, node->input(1));

            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          [](const Node* node) -> bool {
            if (!isInputNonSizeZeroTensor(node)) {
              return false;
            }
            if (node->inputs()[1]->node()->kind() != prim::Constant) {
              return false;
            }
            // we do not support explicit memory_format on output
            if (!node->inputs()[2]->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              return false;
            }
            // we do not support explicit memory_format on output
            if (!node->inputs()[3]->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              return false;
            }
            // we do not support explicit memory_format on output
            if (!node->inputs()[4]->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              return false;
            }
            // we do not support explicit memory_format on output
            if (!node->inputs()[6]->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              return false;
            }
            return true;
          },
          nullptr);
    }

    // Limiting aten::to implementation to only change the dtype of a tensor
    {
      auto ptr_op = getOperatorForLiteral(
          "aten::to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt, value_map[node->inputs()[0]->unique()]);
            auto self = list_val.front();
            list_val.pop_front();

            auto out = castTensoToDtype(self, node->input(1));

            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          [](const Node* node) -> bool {
            if (!isInputNonSizeZeroTensor(node)) {
              return false;
            }
            if (node->inputs()[1]->node()->kind() != prim::Constant) {
              return false;
            }
            // we do not support explicit memory_format on output
            if (!node->inputs()[4]->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              return false;
            }
            return true;
          },
          nullptr);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::type_as(Tensor self, Tensor other) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt, value_map[node->inputs()[0]->unique()]);
            auto self = list_val.front();
            list_val.pop_front();

            // TODO: switch to PyTorch dtype as it's closer to truth.
            // For now, reality is that PyTorch IR profiling information could
            // be missing even with profiling executor, due to upstream
            // transformations between profiling runs to fusion pass.
            auto opt_dtype =
                value_map[node->inputs()[1]->unique()]->getDataType();
            TORCH_INTERNAL_ASSERT(opt_dtype.has_value());

            auto out = castOp(opt_dtype.value(), self);
            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    {
      // We are not fusing `linear` yet, because we can't codegen efficient gemm
      // However, we still need this here, so PE would insert profile node for
      // this node.
      // During fusion pass, We decompose linear into gemm + elementwise.
      auto ptr_op = getOperatorForLiteral(
          "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            // this entry is created so we do profile input tensors;
            TORCH_INTERNAL_ASSERT(false, "not implemented yet");
          },
          [](const Node* node) -> bool {
            // We only profile `linear` layer but not fusing it.
            return false;
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "prim::add_optional(Tensor(a) input, Tensor? bias) -> Tensor(a)");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            // this entry is created so we do profile input tensors;
            if (node->input(1)->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              // forwarding the value;
              value_map.emplace(
                  node->output()->unique(),
                  value_map[node->inputs()[0]->unique()]);
            } else {
              MemoryFormat format;
              std::list<Val*> list_val;
              std::tie(format, list_val) = getPWFormatValues(
                  c10::nullopt,
                  value_map[node->inputs()[0]->unique()],
                  value_map[node->inputs()[1]->unique()]);
              auto lhs = list_val.front();
              list_val.pop_front();
              auto rhs = list_val.front();
              list_val.pop_front();

              auto out = binaryOp(
                  BinaryOpType::Add,
                  lhs,
                  rhs,
                  TypePromotion::default_op_config);
              value_map.emplace(
                  node->output()->unique(), ValueHolder(out, format));
            }
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt, value_map[node->inputs()[0]->unique()]);
            auto self = list_val.front()->as<TensorView>();
            list_val.pop_front();

            Val* negative_slope = value_map[node->inputs()[1]->unique()];

            auto out = leaky_relu(self, negative_slope);
            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          [](const Node* node) -> bool {
            if (!isInputNonSizeZeroTensor(node)) {
              return false;
            }
            return true;
          },
          nullptr);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::gelu(Tensor self, *, str approximate='none') -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt, value_map[node->inputs()[0]->unique()]);
            auto self = list_val.front()->as<TensorView>();
            list_val.pop_front();

            auto approximate = constant_as<std::string>(node->input(1));
            TORCH_INTERNAL_ASSERT(
                approximate.has_value(),
                "The approximate parameter is required.");
            const auto kTanhGelu =
                at::native::get_gelutype_enum(approximate.value()) ==
                at::native::GeluType::Tanh;

            auto out = (kTanhGelu) ? tanh_gelu(self) : gelu(self);
            value_map.emplace(
                node->output()->unique(), ValueHolder(out, format));
          },
          [](const Node* node) -> bool {
            if (!isInputNonSizeZeroTensor(node)) {
              return false;
            }
            if (node->input(1)->node()->kind() != prim::Constant) {
              return false;
            }
            return true;
          },
          nullptr);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::gelu_backward(Tensor grad_output, Tensor self, *, str approximate='none') -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getPWFormatValues(
                c10::nullopt,
                value_map[node->inputs()[0]->unique()],
                value_map[node->inputs()[1]->unique()]);
            auto grad_out = list_val.front()->as<TensorView>();
            list_val.pop_front();
            auto self = list_val.front()->as<TensorView>();
            list_val.pop_front();

            auto approximate = constant_as<std::string>(node->input(2));
            TORCH_INTERNAL_ASSERT(
                approximate.has_value(),
                "The approximate parameter is required.");
            const auto kTanhGelu =
                at::native::get_gelutype_enum(approximate.value()) ==
                at::native::GeluType::Tanh;

            auto grad_in = (kTanhGelu) ? tanh_gelu_backward(grad_out, self)
                                       : gelu_backward(grad_out, self);
            value_map.emplace(
                node->output()->unique(), ValueHolder(grad_in, format));
          },
          [](const Node* node) -> bool {
            if (!isInputNonSizeZeroTensor(node)) {
              return false;
            }
            if (node->input(2)->node()->kind() != prim::Constant) {
              return false;
            }
            return true;
          },
          nullptr);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getPWFormatValues(
                c10::nullopt,
                value_map[node->inputs()[0]->unique()],
                value_map[node->inputs()[1]->unique()]);
            auto grad_out = list_val.front()->as<TensorView>();
            list_val.pop_front();
            auto self = list_val.front()->as<TensorView>();
            list_val.pop_front();

            auto grad_in = tanh_backward(grad_out, self);
            value_map.emplace(
                node->output()->unique(), ValueHolder(grad_in, format));
          },
          isInputNonSizeZeroTensor,
          nullptr);
    }

    {
      std::array<const char*, kNumAminAmaxOps> BinaryFloatOp = {
          "aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor",
          "aten::amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor"};
      for (auto signature : BinaryFloatOp) {
        auto ptr_op = getOperatorForLiteral(signature);
        REGISTER_PARSE_RULE(
            ptr_op,
            {
              MemoryFormat format;
              std::list<Val*> list_val;
              std::tie(format, list_val) = getConsistentValues(
                  MemoryFormat::Contiguous(),
                  value_map[node->inputs()[0]->unique()]);
              auto self = list_val.front();
              list_val.pop_front();
              auto dims_list = constant_as<c10::List<int64_t>>(node->input(1));
              TORCH_INTERNAL_ASSERT(
                  dims_list.has_value(),
                  "aten::amax/amin cannot be fused with dynamic axes");
              std::vector<int> dims;
              if (!dims_list->empty()) {
                for (const auto dim : dims_list->vec()) {
                  dims.emplace_back(static_cast<int>(dim));
                }
              } else {
                dims.resize(self->as<TensorView>()->nDims());
                std::iota(dims.begin(), dims.end(), 0);
              }
              auto keepdim = constant_as<bool>(node->input(2));
              TORCH_INTERNAL_ASSERT(
                  keepdim.has_value(),
                  "aten::amax/amin cannot be fused with dynamic keepdim");

              TensorView* out = nullptr;
              if (node->kind() == c10::Symbol::fromQualString("aten::amax")) {
                out = max(self->as<TensorView>(), dims, keepdim.value());
              } else if (
                  node->kind() == c10::Symbol::fromQualString("aten::amin")) {
                out = min(self->as<TensorView>(), dims, keepdim.value());
              } else {
                TORCH_INTERNAL_ASSERT(
                    false, "unrecognized operation in aten::amax/amin");
              }
              value_map.emplace(node->output()->unique(), out);
            },
            [](const Node* node) -> bool {
              if (isReductionNonCompatibleTensor(
                      node->input(0)->type()->cast<TensorType>())) {
                return false;
              }
              // we don't support dynamic reduction axes;
              if (node->inputs()[1]->node()->kind() != prim::Constant) {
                return false;
              }
              // we don't support dynamic keepdim yet;
              if (node->inputs()[2]->node()->kind() != prim::Constant) {
                return false;
              }
              return true;
            },
            [](const Node* node) -> OperatorType {
              return OperatorType::Reduction;
            });
      }
    }

    {
      std::array<const char*, kNumViewOps> ViewOps = {
          "prim::reshape_copy(Tensor self, int[] shape) -> Tensor",
          "prim::view_copy(Tensor self, int[] size) -> Tensor"};
      for (auto signature : ViewOps) {
        auto ptr_op = getOperatorForLiteral(signature);
        REGISTER_PARSE_RULE(
            ptr_op,
            {
              auto self_value = node->inputs()[0];
              MemoryFormat format;
              std::list<Val*> list_val;
              std::tie(format, list_val) = getConsistentValues(
                  MemoryFormat::Contiguous(), value_map[self_value->unique()]);
              auto self = list_val.front()->as<TensorView>();
              list_val.pop_front();

              auto self_type = self_value->type()->cast<c10::TensorType>();
              TORCH_INTERNAL_ASSERT(self_type != nullptr);
              auto self_sizes = getTensorSizes(self_type);

              auto view_sizes = constant_as<c10::List<int64_t>>(node->input(1));
              TORCH_INTERNAL_ASSERT(
                  view_sizes.has_value(), "The size parameter is required.");

              auto output = view(self, self_sizes, view_sizes->vec());
              value_map.emplace(node->output()->unique(), output);
            },
            [](const Node* node) -> bool {
              auto self_value = node->inputs()[0];
              auto tensor_type = self_value->type()->cast<c10::TensorType>();
              if (tensor_type == nullptr) {
                return false;
              }
              if (!tensor_type->sizes().concrete_sizes().has_value()) {
                // Shape information for input tensor is required.
                return false;
              }

              if (!isInputNonSizeZeroTensor(node)) {
                return false;
              }
              // Reject fusing node if view_sizes contains an inferred dimension
              auto view_sizes = constant_as<c10::List<int64_t>>(node->input(1));
              if (!view_sizes.has_value()) {
                // The size parameter is required.
                return false;
              }

              for (auto axis_size : view_sizes->vec()) {
                if (axis_size == -1) {
                  return false;
                }
              }
              return true;
            },
            nullptr);
      }
    }

    {
      auto flatten_op = getOperatorForLiteral(
          "prim::flatten_copy(Tensor self, int start_dim, int end_dim) -> Tensor");
      REGISTER_PARSE_RULE(
          flatten_op,
          {
            auto self_value = node->inputs()[0];
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                MemoryFormat::Contiguous(), value_map[self_value->unique()]);
            auto self = list_val.front()->as<TensorView>();
            list_val.pop_front();

            auto start_dim_value = constant_as<int>(node->input(1));
            TORCH_INTERNAL_ASSERT(
                start_dim_value.has_value(), "start_dim is not valid");
            auto end_dim_value = constant_as<int>(node->input(2));
            TORCH_INTERNAL_ASSERT(
                end_dim_value.has_value(), "end_dim is not valid");

            TensorView* output =
                flatten(self, start_dim_value.value(), end_dim_value.value());
            value_map.emplace(node->output()->unique(), output);
          },
          [](const Node* node) -> bool {
            // we don't support dynamic start_dim;
            if (node->inputs()[1]->node()->kind() != prim::Constant) {
              return false;
            }
            // we don't support dynamic end_dim yet;
            if (node->inputs()[2]->node()->kind() != prim::Constant) {
              return false;
            }
            return true;
          },
          nullptr);
    }

    {
      auto ptr_op =
          getOperatorForLiteral("prim::squeeze_copy(Tensor self) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            auto self_value = node->inputs()[0];
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                MemoryFormat::Contiguous(), value_map[self_value->unique()]);
            auto self = list_val.front()->as<TensorView>();
            list_val.pop_front();

            auto self_type = self_value->type()->cast<c10::TensorType>();
            TORCH_INTERNAL_ASSERT(self_type != nullptr);
            auto self_sizes = getTensorSizes(self_type);

            TensorView* output = nullptr;
            if (self_sizes.empty()) {
              // squeeze on scalar tensor should just return itself;
              output = set(self);
            } else {
              output = squeeze(self, self_sizes);
            }
            value_map.emplace(node->output()->unique(), output);
          },
          [](const Node* node) -> bool {
            // Shape information for input tensor is required.
            auto self_value = node->inputs()[0];
            auto tensor_type = self_value->type()->cast<c10::TensorType>();
            if (tensor_type == nullptr) {
              return false;
            }
            if (!isInputNonSizeZeroTensor(node)) {
              return false;
            }
            return tensor_type->sizes().concrete_sizes().has_value();
          },
          nullptr);
    }

    {
      std::array<const char*, kNumAliasDimOps> AliasOpWithDim = {
          "prim::squeeze_copy.dim(Tensor self, int dim) -> Tensor",
          "prim::unsqueeze_copy(Tensor self, int dim) -> Tensor"};
      for (auto signature : AliasOpWithDim) {
        auto ptr_op = getOperatorForLiteral(signature);
        REGISTER_PARSE_RULE(
            ptr_op,
            {
              auto self_value = node->inputs()[0];
              MemoryFormat format;
              std::list<Val*> list_val;
              std::tie(format, list_val) = getConsistentValues(
                  MemoryFormat::Contiguous(),
                  value_map[node->inputs()[0]->unique()]);
              auto self = list_val.front()->as<TensorView>();
              list_val.pop_front();

              auto dim_value = constant_as<int>(node->input(1));
              TORCH_INTERNAL_ASSERT(dim_value.has_value(), "dim is not valid");

              TensorView* output = nullptr;
              if (node->kind() == prim::unsqueeze_copy) {
                output = unsqueeze(self, dim_value.value());
              } else {
                auto self_type = self_value->type()->cast<c10::TensorType>();
                TORCH_INTERNAL_ASSERT(self_type != nullptr);
                auto self_sizes = getTensorSizes(self_type);
                if (self_sizes.empty()) {
                  // squeeze on scalar tensor should just return itself;
                  output = set(self);
                } else {
                  output = squeeze(self, self_sizes, dim_value.value());
                }
              }
              value_map.emplace(node->output()->unique(), output);
            },
            [](const Node* node) -> bool {
              // Shape information for input tensor is required.
              auto self_value = node->inputs()[0];
              auto tensor_type = self_value->type()->cast<c10::TensorType>();
              if (tensor_type == nullptr) {
                return false;
              }
              if (!isInputNonSizeZeroTensor(node)) {
                return false;
              }
              if (node->input(1)->node()->kind() != prim::Constant) {
                return false;
              }
              auto optional_sizes = tensor_type->sizes().concrete_sizes();
              return tensor_type->sizes().concrete_sizes().has_value();
            },
            nullptr);
      }
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "prim::expand_as_copy(Tensor self, Tensor other) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getPWFormatValues(
                c10::nullopt,
                value_map[node->inputs()[0]->unique()],
                value_map[node->inputs()[1]->unique()]);
            auto self = list_val.front()->as<TensorView>();
            list_val.pop_front();
            auto other = list_val.front()->as<TensorView>();
            list_val.pop_front();

            auto output = expand_as(self, other);
            value_map.emplace(
                node->output()->unique(), ValueHolder(output, format));
          },
          [](const Node* node) -> bool {
            if (!isInputNonSizeZeroTensor(node)) {
              return false;
            }

            return true;
          },
          nullptr);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "prim::expand_copy(Tensor self, int[] size, *, bool implicit=False) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            auto self_value = node->inputs()[0];
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                MemoryFormat::Contiguous(), value_map[self_value->unique()]);
            auto self = list_val.front()->as<TensorView>();
            list_val.pop_front();

            auto expand_sizes = constant_as<c10::List<int64_t>>(node->input(1));
            TORCH_INTERNAL_ASSERT(
                expand_sizes.has_value(), "The size parameter is required.");

            std::vector<CgValue> expand_sizes_vec;
            for (const int64_t& size : expand_sizes.value()) {
              expand_sizes_vec.push_back(IrBuilder::create<Int>(size));
            }

            // TODO: we should be able to support dynamic expand values
            auto output = expand(self, expand_sizes_vec);
            value_map.emplace(node->output()->unique(), output);
          },
          [](const Node* node) -> bool {
            if (!isInputNonSizeZeroTensor(node)) {
              return false;
            }
            // expand_sizes needs to be constant
            auto expand_sizes = constant_as<c10::List<int64_t>>(node->input(1));
            if (!expand_sizes.has_value()) {
              return false;
            }

            return true;
          },
          nullptr);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "prim::permute_copy.int(Tensor(a) self, int[] dims) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt, value_map[node->inputs()[0]->unique()]);
            auto self_t = list_val.front();
            list_val.pop_front();
            auto self = self_t->as<TensorView>();

            auto dims = constant_as<c10::List<int64_t>>(node->input(1));
            TORCH_INTERNAL_ASSERT(
                dims.has_value(), "The dims parameter is required.");
            TORCH_INTERNAL_ASSERT(
                dims.value().size() == self->getMaybeRFactorDomain().size());

            auto output = permute(self, dims->vec());
            value_map.emplace(
                node->output()->unique(), ValueHolder(output, format));
          },
          [](const Node* node) -> bool {
            if (!isInputNonSizeZeroTensor(node)) {
              return false;
            }
            auto dims = constant_as<c10::List<int64_t>>(node->input(1));
            if (!dims.has_value()) {
              return false;
            }

            return true;
          },
          nullptr);
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "prim::transpose_copy.int(Tensor(a) self, int dim0, int dim1) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt, value_map[node->inputs()[0]->unique()]);
            auto self_t = list_val.front();
            list_val.pop_front();
            auto self = self_t->as<TensorView>();

            auto dim0 = constant_as<int>(node->input(1));
            TORCH_INTERNAL_ASSERT(
                dim0.has_value(), "dim0 in transpose is not valid.");

            auto dim1 = constant_as<int>(node->input(2));
            TORCH_INTERNAL_ASSERT(
                dim1.has_value(), "dim1 in transpose is not valid.");

            auto output = transpose(self, dim0.value(), dim1.value());
            value_map.emplace(
                node->output()->unique(), ValueHolder(output, format));
          },
          [](const Node* node) -> bool {
            if (!isInputNonSizeZeroTensor(node)) {
              return false;
            }
            if (node->input(1)->node()->kind() != prim::Constant) {
              return false;
            }
            if (node->input(2)->node()->kind() != prim::Constant) {
              return false;
            }
            return true;
          },
          nullptr);
    }

    {
      auto ptr_op =
          getOperatorForLiteral("prim::t_copy(Tensor(a) self) -> Tensor");
      REGISTER_PARSE_RULE(
          ptr_op,
          {
            MemoryFormat format;
            std::list<Val*> list_val;
            std::tie(format, list_val) = getConsistentValues(
                c10::nullopt, value_map[node->inputs()[0]->unique()]);
            auto self_t = list_val.front();
            list_val.pop_front();
            auto self = self_t->as<TensorView>();

            TORCH_INTERNAL_ASSERT(self->getMaybeRFactorDomain().size() <= 2);

            auto output = transpose(self);
            value_map.emplace(
                node->output()->unique(), ValueHolder(output, format));
          },
          [](const Node* node) -> bool {
            if (!isInputNonSizeZeroTensor(node)) {
              return false;
            }

            return true;
          },
          nullptr);
    }
  }

  void processJitNode(const JitOp* node) {
    if (node->kind() == prim::Constant) {
      // partition doesn't take constant node explicitly, but it does and copy
      // constant into subgraph. So we need to register constants in codegen IR;
      for (auto output : node->outputs()) {
        TORCH_INTERNAL_ASSERT(
            registerScalar(output),
            "registration of output failed at index ",
            output->offset(),
            " for node ",
            *node);
      }
    } else {
      auto reg_entry = lookupInRegistry(node);
      TORCH_INTERNAL_ASSERT(
          reg_entry != nullptr,
          "CudaFusionGroup Parser doesn't handle node: ",
          canonicalSchemaString(node->schema()));
      reg_entry->parse(node, value_map_);
    }
  }

  bool registerValue(const JitValue* val) {
    return registerInputTensor(val) || registerScalar(val);
  }

  bool registerScalar(const JitValue* val) {
    if (val->type()->isSubtypeOf(
            static_cast<c10::TypePtr>(ComplexType::get()))) {
      CgValue cg_val = nullptr;
      if (auto ival = constant_as<c10::complex<double>>(val)) {
        cg_val = IrBuilder::create<ComplexDouble>(ival.value());
      } else {
        cg_val = IrBuilder::create<ComplexDouble>();
      }
      value_map_.emplace(val->unique(), cg_val);
      return true;
    } else if (val->type()->isSubtypeOf(
                   static_cast<c10::TypePtr>(FloatType::get()))) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      CgValue cg_val;
      if (auto ival = constant_as<double>(val)) {
        cg_val = IrBuilder::create<Double>(ival.value());
      } else {
        cg_val = IrBuilder::create<Double>();
      }
      value_map_.emplace(val->unique(), cg_val);
      return true;
    } else if (val->type()->isSubtypeOf(
                   static_cast<c10::TypePtr>(IntType::get()))) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      CgValue cg_val;
      if (auto ival = constant_as<int64_t>(val)) {
        cg_val = IrBuilder::create<Int>(ival.value());
      } else {
        cg_val = IrBuilder::create<Int>();
      }
      value_map_.emplace(val->unique(), cg_val);
      return true;
    } else if (val->type()->isSubtypeOf(
                   static_cast<c10::TypePtr>(BoolType::get()))) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      CgValue cg_val;
      if (auto ival = constant_as<bool>(val)) {
        cg_val = IrBuilder::create<Bool>(ival.value());
      } else {
        cg_val = IrBuilder::create<Bool>();
      }
      value_map_.emplace(val->unique(), cg_val);
      return true;
    } else if (
        val->type()->isSubtypeOf(
            static_cast<c10::TypePtr>(StringType::get())) ||
        val->type()->isSubtypeOf(
            static_cast<c10::TypePtr>(DeviceObjType::get())) ||
        val->type()->isSubtypeOf(static_cast<c10::TypePtr>(NoneType::get()))) {
      // TODO: should we consider adding support for NoneType;
      // Note: String/Device scalars are only used in parsing rules, do not
      // register string with codegen IR.
      return true;
    } else if (val->type()->cast<ListType>()) {
      // TODO: we don't support list type in codegen yet;
      // This is a WAR to allow axes of reduction to be passed as constant list;
      // We simply ignore conversion if the scalar value is a constant;
      auto ivalue = toIValue(val);
      TORCH_INTERNAL_ASSERT(
          ivalue.has_value(),
          "List[T] is not supported as an argument by NvFuser. Use a Constant List.");
      return true;
    }
    return false;
  }

  bool registerInputTensor(const JitValue* val) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    CgValue cg_val;
    // Don't register if we don't support the type
    if (auto tensor_type = val->type()->cast<c10::TensorType>()) {
      if (!tensor_type->scalarType().has_value()) {
        return false;
      }

      if (aten_to_data_type(tensor_type->scalarType().value()) ==
          DataType::Null) {
        return false;
      }

      // check for NHWC contiguous tensor
      TORCH_CHECK(tensor_type->dim().has_value(), "rank missing");
      const auto n_dim = tensor_type->dim().value();

      MemoryFormat format;
      std::vector<int> stride_index;
      for (const auto i : c10::irange(n_dim)) {
        const auto& stride_property_i = tensor_type->stride_properties()[i];
        if (stride_property_i->stride_index_.has_value()) {
          stride_index.emplace_back(stride_property_i->stride_index_.value());
        }
      }

      // only set permutation when all stride_index are available
      if (stride_index.size() == n_dim) {
        format.setPermutation(stride_index);
      }

      // construct permuted tensor_type
      if (format.hasPermutation()) {
        auto opt_s_vec = tensor_type->symbolic_sizes().sizes();
        TORCH_CHECK(opt_s_vec.has_value(), "missing rank of symbolic sizes");
        std::vector<c10::ShapeSymbol> s_vec = opt_s_vec.value();
        // apply permutation
        auto permutation = format.apply();
        for (auto new_axis : c10::irange(permutation.size())) {
          auto old_axis = permutation.at(new_axis);
          s_vec[new_axis] = opt_s_vec.value()[old_axis];
        }

        // copying stride properties because we need to permute it
        auto opt_stride_vec = tensor_type->stride_properties().sizes();
        TORCH_CHECK(opt_stride_vec.has_value(), "missing stride properties");
        auto nhwc_stride_vec = opt_stride_vec.value();
        // Make tensor contiguous after permutation.
        // Note that we are only updating stride_properties.stride_index, since
        // contiguous_ and stride_ value should remain the same after
        // permutation
        for (const auto i : c10::irange(n_dim)) {
          nhwc_stride_vec[i]->stride_index_ = n_dim - i - 1;
        }

        tensor_type = c10::TensorType::create(
            tensor_type->scalarType(),
            tensor_type->device(),
            s_vec,
            nhwc_stride_vec,
            tensor_type->requires_grad(),
            tensor_type->undefined());
      }

      cg_val = IrBuilder::create<TensorView>(tensor_type);
      if (is_cpu_scalar(*tensor_type)) {
        cg_val->as<TensorView>()->setCpuScalar(true);
      }
      value_map_.emplace(val->unique(), ValueHolder(cg_val, format));
      return true;
    }
    return false;
  }

  std::shared_ptr<Graph> graph_;

  // maps from JitValue::unique() to fusion Val;
  std::unordered_map<size_t, ValueHolder> value_map_;

  static std::unordered_set<Symbol> parser_symbol_set_;
  static std::unordered_set<Symbol> parser_skip_set_;
  static std::mutex parser_mutex_;

  // parsing rule registry.
  static std::unordered_map<std::string, RegistrationEntry>
      jit_operator_registry_; // NOLINT

  // pointing cached entry stored in `jit_operator_registry_`
  static std::unordered_map<const FunctionSchema*, const RegistrationEntry*>
      cached_registry_lookup_; // NOLINT

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static c10::once_flag once_flag_;
};
std::unordered_set<Symbol> IrParser::parser_symbol_set_; // NOLINT
std::unordered_set<Symbol> IrParser::parser_skip_set_; // NOLINT
std::mutex IrParser::parser_mutex_;
std::unordered_map<std::string, IrParser::RegistrationEntry>
    IrParser::jit_operator_registry_; // NOLINT
std::unordered_map<const FunctionSchema*, const IrParser::RegistrationEntry*>
    IrParser::cached_registry_lookup_; // NOLINT

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
c10::once_flag IrParser::once_flag_;

ProfileIValueOp* insertProfileIValueOp(
    Node* node,
    size_t offset,
    ProfilingRecord* pr) {
  auto in_val = node->input(offset);
  auto pn = pr->createProfileIValueNode(in_val);
  pn->insertBefore(node);
  node->replaceInput(offset, pn->output());
  return pn;
}

void profileReductionSize(ProfilingRecord* pr, Node* node, size_t offset) {
  auto pn = insertProfileIValueOp(node, offset, pr);

  const auto ivalue_profiler = [pr, pn](Stack& stack) {
    std::lock_guard<std::mutex> lock(pr->mutex_);

    // TODO: we don't care about merging multiple profiling runs as we don't
    // support it at all;
    int64_t frame_id = 0;
    pop(stack, frame_id);
    IValue value;
    pop(stack, value);

    std::vector<int64_t> size_vec;
    if (value.isIntList()) {
      size_vec = value.toIntVector();
    } else if (value.isNone()) {
      size_vec.clear();
    } else {
      TORCH_INTERNAL_ASSERT(
          false,
          "profileReductionSize does not support data type: ",
          value.tagKind());
    }
    // We stop profiling when it has failed
    if (!pn->hasAttribute(profileFailedAttr)) {
      if (!pn->hasAttribute(reductionSizeAttr)) {
        pn->is_(reductionSizeAttr, size_vec);
      } else {
        auto profiled_ints = pn->is(reductionSizeAttr);
        if (profiled_ints.size() != size_vec.size() ||
            !std::equal(
                profiled_ints.begin(), profiled_ints.end(), size_vec.begin())) {
          TORCH_WARN_ONCE(
              __FUNCTION__,
              " sees varying value in profiling, ignoring and this should be handled by GUARD logic");
          pn->s_(profileFailedAttr, "varying profile values");
          pn->removeAttribute(reductionSizeAttr);
        }
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          !pn->hasAttribute(reductionSizeAttr),
          "profiled attribute should have been removed when profiling is marked as failed");
    }
    push(stack, value);
  };
  pn->setCallback(ivalue_profiler);
}

void profileViewSize(ProfilingRecord* pr, Node* node, size_t offset) {
  auto pn = insertProfileIValueOp(node, offset, pr);

  const auto ivalue_profiler = [pr, pn](Stack& stack) {
    std::lock_guard<std::mutex> lock(pr->mutex_);

    // TODO: we don't care about merging multiple profiling runs as we don't
    // support it at all;
    int64_t frame_id = 0;
    pop(stack, frame_id);
    IValue value;
    pop(stack, value);
    TORCH_INTERNAL_ASSERT(
        value.isIntList(), "profiling seeing the wrong data type");
    if (!pn->hasAttribute(profileFailedAttr)) {
      if (!pn->hasAttribute(viewSizeAttr)) {
        pn->is_(viewSizeAttr, value.toIntVector());
      } else {
        auto profiled_ints = pn->is(viewSizeAttr);
        auto input_ints = value.toIntList();
        if (profiled_ints.size() != input_ints.size() ||
            !std::equal(
                profiled_ints.begin(),
                profiled_ints.end(),
                input_ints.begin())) {
          TORCH_WARN_ONCE(
              __FUNCTION__,
              " sees varying value in profiling, ignoring and this should be handled by GUARD logic");
          pn->s_(profileFailedAttr, "varying profile values");
          pn->removeAttribute(viewSizeAttr);
        }
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          !pn->hasAttribute(viewSizeAttr),
          "profiled attribute should have been removed when profiling is marked as failed");
    }
    push(stack, value);
  };

  pn->setCallback(ivalue_profiler);
}

void profileIntList(ProfilingRecord* pr, Node* node, size_t offset) {
  auto pn = insertProfileIValueOp(node, offset, pr);

  const auto ivalue_profiler = [pr, pn](Stack& stack) {
    std::lock_guard<std::mutex> lock(pr->mutex_);

    // TODO: we don't care about merging multiple profiling runs as we don't
    // support it at all;
    int64_t frame_id = 0;
    pop(stack, frame_id);
    IValue value;
    pop(stack, value);
    TORCH_INTERNAL_ASSERT(
        value.isIntList(), "profiling seeing the wrong data type");
    if (!pn->hasAttribute(profileFailedAttr)) {
      if (!pn->hasAttribute(intListAttr)) {
        pn->is_(intListAttr, value.toIntVector());
      } else {
        auto profiled_ints = pn->is(intListAttr);
        auto input_ints = value.toIntList();
        if (profiled_ints.size() != input_ints.size() ||
            !std::equal(
                profiled_ints.begin(),
                profiled_ints.end(),
                input_ints.begin())) {
          TORCH_WARN_ONCE(
              __FUNCTION__,
              " sees varying value in profiling, ignoring and this should be handled by GUARD logic");
          pn->s_(profileFailedAttr, "varying profile values");
          pn->removeAttribute(intListAttr);
        }
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          !pn->hasAttribute(intListAttr),
          "profiled attribute should have been removed when profiling is marked as failed");
    }
    push(stack, value);
  };

  pn->setCallback(ivalue_profiler);
}

void profileString(ProfilingRecord* pr, Node* node, size_t offset) {
  auto pn = insertProfileIValueOp(node, offset, pr);

  const auto ivalue_profiler = [pr, pn](Stack& stack) {
    std::lock_guard<std::mutex> lock(pr->mutex_);

    // TODO: we don't care about merging multiple profiling runs as we don't
    // support it at all;
    int64_t frame_id = 0;
    pop(stack, frame_id);
    IValue value;
    pop(stack, value);
    TORCH_INTERNAL_ASSERT(
        value.isString(), "profiling seeing the wrong data type");
    if (!pn->hasAttribute(profileFailedAttr)) {
      if (!pn->hasAttribute(strAttr)) {
        pn->s_(strAttr, value.toStringRef());
      } else {
        const auto& profiled_str = pn->s(strAttr);
        const auto& input_str = value.toStringRef();
        if (input_str != profiled_str) {
          TORCH_WARN_ONCE(
              __FUNCTION__,
              " sees varying value in profiling, ignoring and this should be handled by GUARD logic");
          pn->s_(profileFailedAttr, "varying profile values");
          pn->removeAttribute(strAttr);
        }
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          !pn->hasAttribute(strAttr),
          "profiled attribute should have been removed when profiling is marked as failed");
    }
    push(stack, value);
  };

  pn->setCallback(ivalue_profiler);
}

void profileBool(ProfilingRecord* pr, Node* node, size_t offset) {
  auto pn = insertProfileIValueOp(node, offset, pr);

  const auto ivalue_profiler = [pr, pn](Stack& stack) {
    std::lock_guard<std::mutex> lock(pr->mutex_);

    // TODO: we don't care about merging multiple profiling runs as we don't
    // support it at all;
    int64_t frame_id = 0;
    pop(stack, frame_id);
    IValue value;
    pop(stack, value);
    TORCH_INTERNAL_ASSERT(
        value.isBool(), "profiling seeing the wrong data type");
    if (!pn->hasAttribute(profileFailedAttr)) {
      if (!pn->hasAttribute(boolAttr)) {
        pn->i_(boolAttr, value.toBool());
      } else {
        auto profiled_bool = pn->i(boolAttr);
        auto input_bool = value.toBool();
        if (input_bool != profiled_bool) {
          TORCH_WARN_ONCE(
              __FUNCTION__,
              " sees varying value in profiling, ignoring and this should be handled by GUARD logic");
          pn->s_(profileFailedAttr, "varying profile values");
          pn->removeAttribute(boolAttr);
        }
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          !pn->hasAttribute(boolAttr),
          "profiled attribute should have been removed when profiling is marked as failed");
    }
    push(stack, value);
  };

  pn->setCallback(ivalue_profiler);
}

void profileInt(ProfilingRecord* pr, Node* node, size_t offset) {
  auto pn = insertProfileIValueOp(node, offset, pr);

  const auto ivalue_profiler = [pr, pn](Stack& stack) {
    std::lock_guard<std::mutex> lock(pr->mutex_);

    // TODO: we don't care about merging multiple profiling runs as we don't
    // support it at all;
    int64_t frame_id = 0;
    pop(stack, frame_id);
    IValue value;
    pop(stack, value);
    TORCH_INTERNAL_ASSERT(
        value.isInt(), "profiling seeing the wrong data type");
    if (!pn->hasAttribute(profileFailedAttr)) {
      if (!pn->hasAttribute(intAttr)) {
        pn->i_(intAttr, value.toInt());
      } else {
        auto profiled_int = pn->i(intAttr);
        auto input_int = value.toInt();
        if (input_int != profiled_int) {
          TORCH_WARN_ONCE(
              __FUNCTION__,
              " sees varying value in profiling, ignoring and this should be handled by GUARD logic");
          pn->s_(profileFailedAttr, "varying profile values");
          pn->removeAttribute(intAttr);
        }
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          !pn->hasAttribute(intAttr),
          "profiled attribute should have been removed when profiling is marked as failed");
    }
    push(stack, value);
  };

  pn->setCallback(ivalue_profiler);
}

// profile ivalue, used for optional arguments
void profileIval(ProfilingRecord* pr, Node* node, size_t offset) {
  auto pn = insertProfileIValueOp(node, offset, pr);

  const auto ivalue_profiler = [pr, pn](Stack& stack) {
    std::lock_guard<std::mutex> lock(pr->mutex_);

    // TODO: we don't care about merging multiple profiling runs as we don't
    // support it at all;
    int64_t frame_id = 0;
    pop(stack, frame_id);
    IValue value;
    pop(stack, value);
    if (!pn->hasAttribute(profileFailedAttr)) {
      if (!pn->hasAttribute(ivalAttr)) {
        pn->ival_(ivalAttr, value);
      } else {
        auto profiled_ival = pn->ival(ivalAttr);
        if (value != profiled_ival) {
          TORCH_WARN_ONCE(
              __FUNCTION__,
              " sees varying value in profiling, ignoring and this should be handled by GUARD logic");
          pn->s_(profileFailedAttr, "varying profile values");
          pn->removeAttribute(ivalAttr);
        }
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          !pn->hasAttribute(ivalAttr),
          "profiled attribute should have been removed when profiling is marked as failed");
    }
    push(stack, value);
  };

  pn->setCallback(ivalue_profiler);
}

void profileBoolList(ProfilingRecord* pr, Node* node, size_t offset) {
  auto pn = insertProfileIValueOp(node, offset, pr);

  const auto ivalue_profiler = [pr, pn](Stack& stack) {
    std::lock_guard<std::mutex> lock(pr->mutex_);

    // TODO: we don't care about merging multiple profiling runs as we don't
    // support it at all;
    int64_t frame_id = 0;
    pop(stack, frame_id);
    IValue value;
    pop(stack, value);
    TORCH_INTERNAL_ASSERT(
        value.isBoolList(), "profiling seeing the wrong data type");
    if (!pn->hasAttribute(profileFailedAttr)) {
      if (!pn->hasAttribute(boolListAttr)) {
        auto list = value.toBoolList();
        std::vector<int64_t> val(list.begin(), list.end());
        pn->is_(boolListAttr, val);
      } else {
        auto profiled_ints = pn->is(boolListAttr);
        auto input_bools = value.toBoolList();
        if (profiled_ints.size() != input_bools.size() ||
            !std::equal(
                input_bools.begin(),
                input_bools.end(),
                profiled_ints.begin())) {
          TORCH_WARN_ONCE(
              __FUNCTION__,
              " sees varying value in profiling, ignoring and this should be handled by GUARD logic");
          pn->s_(profileFailedAttr, "varying profile values");
          pn->removeAttribute(boolListAttr);
        }
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          !pn->hasAttribute(boolListAttr),
          "profiled attribute should have been removed when profiling is marked as failed");
    }
    push(stack, value);
  };

  pn->setCallback(ivalue_profiler);
}

bool anyInBlock(
    const Block* block,
    const std::function<bool(const Node*)>& fn) {
  for (auto node : block->nodes()) {
    if (fn(node)) {
      return true;
    }
    for (auto block : node->blocks()) {
      if (anyInBlock(block, fn)) {
        return true;
      }
    }
  }
  return false;
}

} // namespace

bool hasReductionNode(const Block* block) {
  return anyInBlock(block, isReductionNode);
}

bool isReductionNode(const Node* node) {
  return IrParser::isReductionNode(node);
}

bool isReductionToSizeNode(const Node* node) {
  return IrParser::isReductionToSizeNode(node);
}

bool hasNormalizationNode(const Block* block) {
  return anyInBlock(block, isNormalizationNode);
}

bool isNormalizationNode(const Node* node) {
  return IrParser::isNormalizationNode(node);
}

bool isElementWiseNode(const Node* node) {
  return IrParser::isElementWiseNode(node);
}

bool isNodeParsible(const Node* node) {
  return IrParser::canParseNode(node);
}

bool shouldProfileNode(const Node* node) {
  return IrParser::lookupInSymbolSet(node);
}

bool skipNodeKind(const std::string& symbol_str, bool flip) {
  return IrParser::querySkipSymbolSet(
      c10::Symbol::fromQualString(symbol_str), flip);
}

bool insertProfileIValue(ProfilingRecord* pr, Node* node, size_t offset) {
  // is skip constant necessary?
  if (node->input(offset)->node()->kind() == prim::Constant) {
    return false;
  }

  static auto dropout_schema =
      getOperatorForLiteral(
          "aten::dropout(Tensor input, float p, bool train) -> Tensor")
          ->schema();
  static auto native_dropout_schema =
      getOperatorForLiteral(
          "aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)")
          ->schema();
  if (node->matches(dropout_schema) || node->matches(native_dropout_schema)) {
    switch (offset) {
      // argument 2: Is training?
      case 2:
        profileBool(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto amax_schema =
      getOperatorForLiteral(
          "aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor")
          ->schema();
  static auto amin_schema =
      getOperatorForLiteral(
          "aten::amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor")
          ->schema();
  if (node->matches(amax_schema) || node->matches(amin_schema)) {
    switch (offset) {
      // argument 1: reduction axes;
      case 1:
        profileIntList(pr, node, offset);
        break;
      // argument 2: keepdim;
      case 2:
        profileBool(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto reduction_operator_schema =
      getOperatorForLiteral(
          "aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)")
          ->schema();
  if (node->matches(reduction_operator_schema)) {
    switch (offset) {
      // argument 1: reduction axes;
      case 1:
        profileIntList(pr, node, offset);
        break;
      // argument 2: keepdim;
      case 2:
        profileBool(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto sum_to_size_schema =
      getOperatorForLiteral(
          "aten::sum_to_size(Tensor self, int[] size) -> Tensor")
          ->schema();
  static auto grad_sum_to_size_schema =
      getOperatorForLiteral(
          "aten::_grad_sum_to_size(Tensor(a) self, int[]? size) -> Tensor(a)")
          ->schema();
  if (node->matches(sum_to_size_schema) ||
      node->matches(grad_sum_to_size_schema)) {
    switch (offset) {
      // argument 1: reduction sizes;
      case 1:
        // TODO(profile_size): double check optional[size]?
        profileReductionSize(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto reshape_schema =
      getOperatorForLiteral("aten::reshape(Tensor self, int[] shape) -> Tensor")
          ->schema();
  static auto reshape_copy_schema =
      getOperatorForLiteral(
          "prim::reshape_copy(Tensor self, int[] shape) -> Tensor")
          ->schema();
  static auto view_schema =
      getOperatorForLiteral("aten::view(Tensor self, int[] size) -> Tensor")
          ->schema();
  static auto view_copy_schema =
      getOperatorForLiteral(
          "prim::view_copy(Tensor self, int[] size) -> Tensor")
          ->schema();
  if (node->matches(reshape_schema) || node->matches(reshape_copy_schema) ||
      node->matches(view_schema) || node->matches(view_copy_schema)) {
    switch (offset) {
      // argument 1: new tensor size;
      case 1:
        profileViewSize(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto flatten_schema1 =
      getOperatorForLiteral(
          "aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor")
          ->schema();
  static auto flatten_schema2 =
      getOperatorForLiteral(
          "prim::flatten_copy(Tensor self, int start_dim, int end_dim) -> Tensor")
          ->schema();
  if (node->matches(flatten_schema1) || node->matches(flatten_schema2)) {
    switch (offset) {
      // argument 1: start_dim;
      // argument 2: end_dim;
      case 1:
      case 2:
        profileInt(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto squeeze_dim_schema =
      getOperatorForLiteral(
          "prim::squeeze_copy.dim(Tensor self, int dim) -> Tensor")
          ->schema();
  static auto unsqueeze_schema =
      getOperatorForLiteral(
          "prim::unsqueeze_copy(Tensor self, int dim) -> Tensor")
          ->schema();
  if (node->matches(squeeze_dim_schema) || node->matches(unsqueeze_schema)) {
    switch (offset) {
      // argument 1: unsqueeze dim;
      case 1:
        profileInt(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto permute_schema =
      getOperatorForLiteral(
          "aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)")
          ->schema();
  static auto permute_copy_schema =
      getOperatorForLiteral(
          "prim::permute_copy(Tensor(a) self, int[] dims) -> Tensor")
          ->schema();
  if (node->matches(permute_schema) || node->matches(permute_copy_schema)) {
    switch (offset) {
      // argument 1: dims;
      case 1:
        profileIntList(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto transpose_int_copy_schema =
      getOperatorForLiteral(
          "aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)")
          ->schema();
  static auto transpose_int_schema =
      getOperatorForLiteral(
          "prim::transpose_copy.int(Tensor(a) self, int dim0, int dim1) -> Tensor")
          ->schema();
  if (node->matches(transpose_int_copy_schema) ||
      node->matches(transpose_int_schema)) {
    switch (offset) {
      // argument 1: dim0;
      // argument 2: dim1;
      case 1:
      case 2:
        profileInt(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto batch_norm_impl_index_schema =
      getOperatorForLiteral(
          "aten::_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, Tensor, int)")
          ->schema();
  static auto native_batch_norm_schema =
      getOperatorForLiteral(
          "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)")
          ->schema();
  static auto batch_norm_schema =
      getOperatorForLiteral(
          "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor")
          ->schema();
  static auto instance_norm_schema =
      getOperatorForLiteral(
          "aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor")
          ->schema();
  if (node->matches(native_batch_norm_schema) ||
      node->matches(batch_norm_impl_index_schema) ||
      node->matches(batch_norm_schema) || node->matches(instance_norm_schema)) {
    switch (offset) {
      // argument 5: training;
      case 5:
        profileBool(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto gelu_schema =
      getOperatorForLiteral(
          "aten::gelu(Tensor self, *, str approximate='none') -> Tensor")
          ->schema();
  if (node->matches(gelu_schema)) {
    switch (offset) {
      // argument 1: approximate;
      case 1:
        profileString(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto gelu_backward_schema =
      getOperatorForLiteral(
          "aten::gelu_backward(Tensor grad_output, Tensor self, *, str approximate='none') -> Tensor")
          ->schema();
  if (node->matches(gelu_backward_schema)) {
    switch (offset) {
      // argument 2: approximate;
      case 2:
        profileString(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto native_layer_norm_schema =
      getOperatorForLiteral(
          "aten::native_layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)")
          ->schema();
  static auto layer_norm_schema =
      getOperatorForLiteral(
          "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor")
          ->schema();
  if (node->matches(native_layer_norm_schema) ||
      node->matches(layer_norm_schema)) {
    switch (offset) {
      case 1:
        profileIntList(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto batch_norm_impl_index_backward_schema =
      getOperatorForLiteral(
          "aten::_batch_norm_impl_index_backward(int impl_index, Tensor input, Tensor grad_output, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var_transform, bool train, float eps, bool[3] output_mask, Tensor reservedSpace) -> (Tensor, Tensor, Tensor)")
          ->schema();
  if (node->matches(batch_norm_impl_index_backward_schema)) {
    switch (offset) {
      // TODO: guard impl_index, but I think that's not needed;
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
      case 8: // argument 8: training;
        profileBool(pr, node, offset);
        break;
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
      case 10:
        profileBoolList(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto batch_norm_backward_schema =
      getOperatorForLiteral(
          "aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")
          ->schema();
  if (node->matches(batch_norm_backward_schema)) {
    switch (offset) {
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
      case 7: // argument 8: training;
        profileBool(pr, node, offset);
        break;
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
      case 9:
        profileBoolList(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto native_layer_norm_backward_schema =
      getOperatorForLiteral(
          "aten::native_layer_norm_backward(Tensor grad_out, Tensor input, int[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")
          ->schema();
  if (node->matches(native_layer_norm_backward_schema)) {
    switch (offset) {
      case 2:
        profileIntList(pr, node, offset);
        break;
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
      case 7:
        profileBoolList(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto to_copy_schema =
      getOperatorForLiteral(
          "aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor")
          ->schema();
  if (node->matches(to_copy_schema)) {
    switch (offset) {
      case 1:
        profileInt(pr, node, offset);
        return true;
      default:
        return false;
    }
  }

  static auto to_dtype_schema =
      getOperatorForLiteral(
          "aten::to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor")
          ->schema();
  if (node->matches(to_dtype_schema)) {
    switch (offset) {
      case 1:
        profileInt(pr, node, offset);
        return true;
      default:
        return false;
    }
  }

  static auto log_softmax_data_schema =
      getOperatorForLiteral(
          "aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor")
          ->schema();
  static auto softmax_data_schema =
      getOperatorForLiteral(
          "aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor")
          ->schema();
  if (node->matches(log_softmax_data_schema) ||
      node->matches(softmax_data_schema)) {
    switch (offset) {
      case 2:
        profileIval(pr, node, offset);
        return true;
      default:
        return false;
    }
  }

  static auto log_softmax_backward_data_schema =
      getOperatorForLiteral(
          "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor")
          ->schema();
  static auto softmax_backward_data_schema =
      getOperatorForLiteral(
          "aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor")
          ->schema();
  if (node->matches(log_softmax_backward_data_schema) ||
      node->matches(softmax_backward_data_schema)) {
    switch (offset) {
      case 2:
        profileInt(pr, node, offset);
        return true;
      case 3:
        profileInt(pr, node, offset);
        return true;
      default:
        return false;
    }
  }

  static auto var_dim_schema =
      getOperatorForLiteral(
          "aten::var.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor")
          ->schema();
  static auto std_dim_schema =
      getOperatorForLiteral(
          "aten::std.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor")
          ->schema();
  if (node->matches(var_dim_schema) || node->matches(std_dim_schema)) {
    switch (offset) {
      case 1:
        profileIntList(pr, node, offset);
        return true;
      case 2:
        profileBool(pr, node, offset);
        return true;
      case 3:
        profileBool(pr, node, offset);
        return true;
      default:
        return false;
    }
  }

  return false;
}

void insertProfileNodesForCUDAFuser_(Block* block, ProfilingRecord* pr) {
  for (const auto& n : block->nodes()) {
    for (const auto offset : c10::irange(n->inputs().size())) {
      insertProfileIValue(pr, n, offset);
    }

    for (auto ib : n->blocks()) {
      insertProfileNodesForCUDAFuser_(ib, pr);
    }
  }
}

void InsertProfileNodes(ProfilingRecord* pr) {
  insertProfileNodesForCUDAFuser_(pr->profiled_graph_->block(), pr);
}

std::unique_ptr<Fusion> parseJitIR(const std::shared_ptr<Graph>& graph) {
  FUSER_PERF_SCOPE("parseJitIR");

  IrParser parser(graph);
  return parser.parse();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
