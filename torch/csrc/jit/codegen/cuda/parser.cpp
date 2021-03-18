#include <torch/csrc/jit/codegen/cuda/parser.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/csrc/jit/ir/constants.h>

#include <unordered_map>
#include <utility>

namespace torch {
namespace jit {

typedef Value JitValue;
typedef Node JitOp;

namespace fuser {
namespace cuda {

constexpr auto kNumUnaryOps = 32;
constexpr auto kNumBinaryOps = 29;
constexpr auto kNumBinaryOpsWithAlpha = 4;
constexpr auto kNumLerpOps = 2;
constexpr auto kNumLayernormFwd = 2;
constexpr auto kNumSumToSize = 2;

namespace {

const auto& sizeAttr = Symbol::attr("profiled_size");
const auto& intListAttr = Symbol::attr("profiled_int_list");
const auto& boolListAttr = Symbol::attr("profiled_bool_list");
const auto& boolAttr = Symbol::attr("profiled_bool");

typedef Val* CgValue;
typedef Expr* CgOp;

typedef void (*ParseFuncPtr)(const Node*, std::unordered_map<size_t, CgValue>&);
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

    void parse(const Node* node, std::unordered_map<size_t, CgValue>& values)
        const {
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
  IrParser(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {
    initRegistry();
  }

  std::unique_ptr<Fusion> parse() {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    auto block = graph_->block();

    // register all inputs;
    for (auto val : block->inputs()) {
      TORCH_INTERNAL_ASSERT(
          registerValue(val),
          "Failure when register value: ",
          *(val->node()),
          " with type: ",
          val->type());
      fusion->addInput(value_map_[val->unique()]);

      auto opt_dtype = value_map_[val->unique()]->getDataType();
      // computation promotion, we cast fp16 inputs to fp32 and use promoted
      // type in the computation.
      if (opt_dtype.has_value() && opt_dtype.value() == DataType::Half) {
        Val* promoted_val = castOp(DataType::Float, value_map_[val->unique()]);
        value_map_[val->unique()] = promoted_val;
      }
    }

    // TODO: disable unroll to ensure rand_like generates identical output as
    // with eager mode
    bool disable_unroll = false;
    bool has_reduction = false;
    // compose nodes in topo order;
    for (const JitOp* node : block->nodes()) {
      processJitNode(node);
      if (node->kind() == aten::rand_like) {
        disable_unroll = true;
      }
      if (node->kind() == aten::sum) {
        has_reduction = true;
      }
    }

    // mark output;
    for (auto jit_output : block->outputs()) {
      TensorView* out = value_map_[jit_output->unique()]->as<TensorView>();
      // demote output dtype to be match PyTorch JIT graph.
      auto tensor_type = jit_output->type()->cast<TensorType>();
      TORCH_INTERNAL_ASSERT(
          tensor_type, "output of fusion group is not TensorType.");
      if (tensor_type->scalarType() == at::ScalarType::Half) {
        // No need to update value_map_ after this point.
        out = castOp(DataType::Half, out)->as<TensorView>();
      }
      fusion->addOutput(out);
    }
    return fusion;
  }

  // return nullptr if entry does not exist
  static const RegistrationEntry* lookupInRegistry(const Node* node) {
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

  static void initRegistry() {
    if (init_registry_) {
      // TODO: mutex this guy;
      registerJitOperator();
      init_registry_ = false;
    }
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
        "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor"};
    for (auto signature : BinaryOpWithAlpha) {
      auto ptr_op = getOperatorForLiteral(signature);
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
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
                          static_cast<BinaryOpWithAlphaType>(&sub_alpha))}});
            // TODO: handle scaling factor when it's not constant 1;
            auto lhs = value_map[node->inputs()[0]->unique()];
            auto rhs = value_map[node->inputs()[1]->unique()];
            auto alpha = value_map[node->inputs()[2]->unique()];

            if (alpha->isOneInt()) {
              auto out = binaryOp(op_mapping[node->kind()].first, lhs, rhs);
              value_map.emplace(node->output()->unique(), out);
            } else {
              auto out = op_mapping[node->kind()].second(lhs, rhs, alpha);
              value_map.emplace(node->output()->unique(), out);
            }
          });
    }

    std::array<const char*, kNumBinaryOps> BinaryOp = {
        "aten::div(Tensor self, Tensor other) -> Tensor",
        "aten::div(Tensor self, Scalar other) -> Tensor",
        "aten::mul(Tensor self, Tensor other) -> Tensor",
        "aten::mul(Tensor self, Scalar other) -> Tensor",
        "aten::atan2(Tensor self, Tensor other) -> Tensor",
        "aten::max(Tensor self, Tensor other) -> Tensor",
        "aten::min(Tensor self, Tensor other) -> Tensor",
        "aten::pow(Tensor self, Tensor exponent) -> Tensor",
        "aten::pow(Tensor self, Scalar exponent) -> Tensor",
        "aten::pow(Scalar self, Tensor exponent) -> Tensor",
        "aten::remainder(Tensor self, Tensor other) -> Tensor",
        "aten::fmod(Tensor self, Tensor other) -> Tensor",
        "aten::__and__(Tensor self, Tensor other) -> Tensor",
        "aten::__or__(Tensor self, Tensor other) -> Tensor",
        "aten::__xor__(Tensor self, Tensor other) -> Tensor",
        "aten::__lshift__(Tensor self, Tensor other) -> Tensor",
        "aten::__rshift__(Tensor self, Tensor other) -> Tensor",
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
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            static std::unordered_map<Symbol, BinaryOpType> op_mapping(
                {{aten::div, BinaryOpType::Div},
                 {aten::mul, BinaryOpType::Mul},
                 {aten::add, BinaryOpType::Add},
                 {aten::sub, BinaryOpType::Sub},
                 {aten::atan2, BinaryOpType::Atan2},
                 {aten::min, BinaryOpType::Min},
                 {aten::max, BinaryOpType::Max},
                 {aten::pow, BinaryOpType::Pow},
                 {aten::remainder, BinaryOpType::Remainder},
                 {aten::fmod, BinaryOpType::Fmod},
                 {aten::lt, BinaryOpType::LT},
                 {aten::le, BinaryOpType::LE},
                 {aten::gt, BinaryOpType::GT},
                 {aten::ge, BinaryOpType::GE},
                 {aten::ne, BinaryOpType::NE},
                 {aten::eq, BinaryOpType::Eq},
                 {aten::__and__, BinaryOpType::And},
                 {aten::__or__, BinaryOpType::Or},
                 {aten::__xor__, BinaryOpType::Xor},
                 {aten::__lshift__, BinaryOpType::Lshift},
                 {aten::__rshift__, BinaryOpType::Rshift}});
            auto lhs = value_map[node->inputs()[0]->unique()];
            auto rhs = value_map[node->inputs()[1]->unique()];

            auto out = binaryOp(op_mapping[node->kind()], lhs, rhs);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    // TODO: cast operations should be merged in.
    std::array<const char*, kNumUnaryOps> UnaryOp = {
        "aten::neg(Tensor self) -> Tensor",
        "aten::abs(Tensor self) -> Tensor",
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
        "aten::tanh(Tensor self) -> Tensor",
        "aten::atan(Tensor self) -> Tensor",
        "aten::sqrt(Tensor self) -> Tensor",
        "aten::rsqrt(Tensor self) -> Tensor",
        "aten::ceil(Tensor self) -> Tensor",
        "aten::floor(Tensor self) -> Tensor",
        "aten::round(Tensor self) -> Tensor",
        "aten::trunc(Tensor self) -> Tensor",
        "aten::bitwise_not(Tensor self) -> Tensor",
        "aten::frac(Tensor self) -> Tensor",
        "aten::reciprocal(Tensor self) -> Tensor",
        "aten::relu(Tensor self) -> Tensor",
        "aten::sigmoid(Tensor self) -> Tensor",
        "aten::gelu(Tensor self) -> Tensor",
    };
    for (auto signature : UnaryOp) {
      auto ptr_op = getOperatorForLiteral(signature);
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            static std::unordered_map<Symbol, UnaryOpType> op_mapping({
                {aten::neg, UnaryOpType::Neg},
                {aten::abs, UnaryOpType::Abs},
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
                {aten::sqrt, UnaryOpType::Sqrt},
                {aten::rsqrt, UnaryOpType::Rsqrt},
                {aten::ceil, UnaryOpType::Ceil},
                {aten::floor, UnaryOpType::Floor},
                {aten::round, UnaryOpType::Round},
                {aten::trunc, UnaryOpType::Trunc},
                {aten::bitwise_not, UnaryOpType::Not},
                {aten::frac, UnaryOpType::Frac},
                {aten::reciprocal, UnaryOpType::Reciprocal},
                {aten::relu, UnaryOpType::Relu},
                {aten::sigmoid, UnaryOpType::Sigmoid},
                {aten::gelu, UnaryOpType::Gelu},
            });
            auto operand = value_map[node->input()->unique()];

            auto out = unaryOp(op_mapping[node->kind()], operand);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            auto operand = value_map[node->inputs()[0]->unique()];

            auto out = unaryOp(UnaryOpType::RandLike, operand);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            auto operand = value_map[node->inputs()[0]->unique()];
            auto th = value_map[node->inputs()[1]->unique()];
            auto value = value_map[node->inputs()[2]->unique()];

            auto out = threshold(operand, th, value);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::clamp(Tensor self, Scalar? min, Scalar? max) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            auto operand = value_map[node->inputs()[0]->unique()];
            // TODO: we need to get a proper lower bound per dtype in operand.
            auto low = value_map.count(node->inputs()[1]->unique()) != 0
                ? value_map[node->inputs()[1]->unique()]
                : new Double(std::numeric_limits<float>::min());
            auto high = value_map.count(node->inputs()[2]->unique()) != 0
                ? value_map[node->inputs()[2]->unique()]
                : new Double(std::numeric_limits<float>::max());

            auto out = clamp(operand, low, high);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::where(Tensor condition, Tensor self, Tensor other) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            auto condition = value_map[node->inputs()[0]->unique()];
            auto x = value_map[node->inputs()[1]->unique()];
            auto y = value_map[node->inputs()[2]->unique()];

            auto out = where(condition, x, y);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    {
      std::array<const char*, kNumLerpOps> LerpOp = {
          "aten::lerp(Tensor self, Tensor end, Scalar weight) -> Tensor",
          "aten::lerp(Tensor self, Tensor end, Tensor weight) -> Tensor"};
      for (auto signature : LerpOp) {
        auto ptr_op = getOperatorForLiteral(signature);
        registerParseRule(
            ptr_op,
            [](const Node* node,
               std::unordered_map<size_t, CgValue>& value_map) -> void {
              auto self = value_map[node->inputs()[0]->unique()];
              auto end = value_map[node->inputs()[1]->unique()];
              auto weight = value_map[node->inputs()[2]->unique()];

              auto out = lerp(self, end, weight);
              value_map.emplace(node->output()->unique(), out);
            });
      }
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            auto self = value_map[node->inputs()[0]->unique()];
            auto tensor1 = value_map[node->inputs()[1]->unique()];
            auto tensor2 = value_map[node->inputs()[2]->unique()];
            auto value = value_map[node->inputs()[3]->unique()];

            auto out = addcmul(self, tensor1, tensor2, value);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::native_dropout(Tensor input, float p, float scale, bool train) -> (Tensor, Tensor)");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            auto input = value_map[node->input(0)->unique()];
            auto prob = value_map[node->input(1)->unique()];
            auto scale = value_map[node->input(2)->unique()];
            auto train = constant_as<bool>(node->input(3));

            TORCH_INTERNAL_ASSERT(
                train.has_value() and train.value(),
                "Train parameter is incorrectly set to false!");

            auto rand_vals = unaryOp(UnaryOpType::RandLike, input);
            auto mask = lt(rand_vals, prob);
            auto apply_mask = mul(input, mask);
            auto out = mul(apply_mask, scale);

            value_map.emplace(node->output(0)->unique(), out);
            value_map.emplace(node->output(1)->unique(), mask);
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::dropout(Tensor input, float p, bool train) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            auto input = value_map[node->input(0)->unique()];
            auto train = constant_as<bool>(node->input(2));
            TORCH_INTERNAL_ASSERT(
                train.has_value(), "dropout needs constant `train` flag");

            if (train.value()) {
              auto prob = value_map[node->input(1)->unique()];
              auto p1m = sub(new Double(1.), prob);

              auto zero_check = add(eq(p1m, new Double(0.)), p1m);
              auto scale = div(new Double(1.), zero_check);
              auto rand_vals = unaryOp(UnaryOpType::RandLike, input);
              auto mask = lt(rand_vals, p1m);
              auto apply_mask = mul(input, mask);
              auto out = mul(apply_mask, scale);

              value_map.emplace(node->output()->unique(), out);
            } else {
              value_map.emplace(node->output()->unique(), input);
            }
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::native_dropout_backward(Tensor grad, Tensor mask, float scale) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            auto grad = value_map[node->input(0)->unique()];
            auto mask = value_map[node->input(1)->unique()];
            auto scale = value_map[node->input(2)->unique()];

            auto temp = mul(grad, mask);
            auto out = mul(temp, scale);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            auto input = value_map[node->input(0)->unique()]->as<TensorView>();

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
            // auto training = constant_as<bool>(node->input(5));
            // TORCH_INTERNAL_ASSERT(
            //     training.has_value(),
            //     "The training (bool) parameter is required.");
            // const bool kTraining = training.value();

            // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
            // auto momentum = constant_as<float>(node->input(6));
            // TORCH_INTERNAL_ASSERT(
            //     momentum.has_value(),
            //     "The momentum (float) parameter is required.");
            // const float kMomentum = momentum.value();

            // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
            auto eps = constant_as<float>(node->input(7));
            TORCH_INTERNAL_ASSERT(
                eps.has_value(), "The EPS parameter is required.");
            const float kEps = eps.value();

            // TODO: NAN when mean and variance are zero
            // --ftz=true -- flush-to-zero

            const size_t kNumberOfDims = input->nDims();
            std::vector<int> reduction_axes;
            std::vector<bool> broadcast_mask(kNumberOfDims, false);
            Val* num_features = new Double(1);
            for (size_t axis = 0; axis < kNumberOfDims; ++axis) {
              if (axis != 1) {
                reduction_axes.push_back(axis);
                broadcast_mask[axis] = true;
                num_features = mul(
                    num_features, input->domain()->domain()[axis]->extent());
              }
            }

            // Algorithm
            auto x_sum = sum(input, reduction_axes);
            auto x_sum_bcast = broadcast(x_sum, broadcast_mask);
            auto x_mean = div(x_sum_bcast, num_features);

            // auto current_mean_hat = mul(x_mean, new Double(kMomentum));
            // auto rmean_bcast = broadcast(running_mean, broadcast_mask);
            // auto mean_hat = mul(rmean_bcast, new Double(1.0 - kMomentum));
            // auto new_mean_hat = add(mean_hat, current_mean_hat);

            auto x_mean_sub = sub(input, x_mean);
            auto x_mean_sub_pow = mul(x_mean_sub, x_mean_sub);
            auto var_sum = sum(x_mean_sub_pow, reduction_axes);
            auto var_sum_bcast = broadcast(var_sum, broadcast_mask);
            auto var = div(var_sum_bcast, num_features);

            // auto num_feature_decrement = sub(num_features, new Int(1));
            // auto unbiased_var = div(var_sum_bcast, num_feature_decrement);
            // auto current_var_hat = mul(unbiased_var, new Double(kMomentum));
            // auto rvar_bcast = broadcast(running_var, broadcast_mask);
            // auto var_hat = mul(rvar_bcast, new Double(1.0 - kMomentum));
            // auto new_var_hat = add(var_hat, current_var_hat);

            auto var_eps = add(var, new Double(kEps));
            auto rvar = unaryOp(UnaryOpType::Rsqrt, var_eps);
            auto output = mul(x_mean_sub, rvar);

            // Optional: norm * weight
            if (weight) {
              auto weight_bcast = broadcast(weight, broadcast_mask);
              output = mul(output, weight_bcast);
            }

            // Optional: norm * weight + bias
            if (bias) {
              auto bias_bcast = broadcast(bias, broadcast_mask);
              output = add(output, bias_bcast);
            }
            value_map.emplace(node->output()->unique(), output);
          },
          [](const Node* node) -> bool { return true; },
          [](const Node* node) -> OperatorType {
            return OperatorType::Normalization;
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            auto input = value_map[node->input(0)->unique()]->as<TensorView>();

            auto norm_shape = constant_as<c10::List<int64_t>>(node->input(1));
            TORCH_INTERNAL_ASSERT(
                norm_shape.has_value(),
                "The Normalized_Shape list is required.");

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

            auto eps = constant_as<float>(node->input(4));
            TORCH_INTERNAL_ASSERT(
                eps.has_value(), "The EPS parameter is required.");
            const float kEps = eps.value();

            const size_t kNormShapeNumDims = norm_shape->vec().size();
            const size_t kOuterNumDims = input->nDims() - kNormShapeNumDims;

            std::vector<int> outer_reduction_axes(kOuterNumDims);
            std::vector<bool> outer_broadcast_mask(input->nDims(), false);
            for (size_t idx = 0; idx < kOuterNumDims; ++idx) {
              outer_reduction_axes[idx] = idx;
              outer_broadcast_mask[idx] = true;
            }

            std::vector<int> inner_reduction_axes(kNormShapeNumDims);
            std::vector<bool> inner_broadcast_mask(input->nDims(), false);
            Val* num_features = new Double(1);
            for (size_t idx = 0; idx < kNormShapeNumDims; ++idx) {
              const size_t axis = input->nDims() - 1 - idx;
              inner_reduction_axes[idx] = axis;
              inner_broadcast_mask[axis] = true;
              num_features =
                  mul(num_features, input->domain()->domain()[axis]->extent());
            }

            // TODO: NAN when mean and variance are zero
            // --ftz=true -- flush-to-zero

            // Algorithm
            auto x_sum = sum(input, inner_reduction_axes);
            auto x_sum_bcast = broadcast(x_sum, inner_broadcast_mask);
            auto x_mean = div(x_sum_bcast, num_features);
            auto x_mean_sub = sub(input, x_mean);
            auto x_mean_sub_pow = mul(x_mean_sub, x_mean_sub);
            auto var_sum = sum(x_mean_sub_pow, inner_reduction_axes);
            auto var_sum_bcast = broadcast(var_sum, inner_broadcast_mask);
            auto var = div(var_sum_bcast, num_features);
            auto var_eps = add(var, new Double(kEps));
            auto rvar = unaryOp(UnaryOpType::Rsqrt, var_eps);
            auto output = mul(x_mean_sub, rvar);

            // Optional: norm * weight
            if (weight) {
              auto weight_bcast = broadcast(weight, outer_broadcast_mask);
              output = mul(output, weight_bcast);
            }

            // Optional: norm * weight + bias
            if (bias) {
              auto bias_bcast = broadcast(bias, outer_broadcast_mask);
              output = add(output, bias_bcast);
            }
            value_map.emplace(node->output()->unique(), output);
          },
          // TODO: #ProfileIValue List should update this
          [](const Node* node) -> bool { return true; },
          [](const Node* node) -> OperatorType {
            return OperatorType::Normalization;
          });
    }

    {
      std::array<const char*, kNumLayernormFwd> LayerNormFwd = {
          "aten::native_layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)",
          "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor"};
      for (auto signature : LayerNormFwd) {
        auto ptr_op = getOperatorForLiteral(signature);
        registerParseRule(
            ptr_op,
            [](const Node* node,
               std::unordered_map<size_t, CgValue>& value_map) -> void {
              auto input =
                  value_map[node->input(0)->unique()]->as<TensorView>();

              auto norm_shape = constant_as<c10::List<int64_t>>(node->input(1));
              TORCH_INTERNAL_ASSERT(
                  norm_shape.has_value(),
                  "The Normalized_Shape list is required.");

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
                eps_ptr = new Double(eps.value());
              } else {
                eps_ptr = value_map[node->input(4)->unique()];
              }

              const size_t kNormShapeNumDims = norm_shape->vec().size();
              const size_t kOuterNumDims = input->nDims() - kNormShapeNumDims;

              std::vector<int> outer_reduction_axes(kOuterNumDims);
              std::vector<bool> outer_broadcast_mask(input->nDims(), false);
              for (size_t idx = 0; idx < kOuterNumDims; ++idx) {
                outer_reduction_axes[idx] = idx;
                outer_broadcast_mask[idx] = true;
              }

              std::vector<int> inner_reduction_axes(kNormShapeNumDims);
              std::vector<bool> inner_broadcast_mask(input->nDims(), false);
              Val* num_features = new Double(1);
              for (size_t idx = 0; idx < kNormShapeNumDims; ++idx) {
                const size_t axis = input->nDims() - 1 - idx;
                inner_reduction_axes[idx] = axis;
                inner_broadcast_mask[axis] = true;
                num_features = mul(
                    num_features, input->domain()->domain()[axis]->extent());
              }

              // TODO: NAN when mean and variance are zero
              // --ftz=true -- flush-to-zero

              // Algorithm
              auto x_sum = sum(input, inner_reduction_axes);
              auto x_sum_bcast = broadcast(x_sum, inner_broadcast_mask);
              auto x_mean = div(x_sum_bcast, num_features);
              auto x_mean_sub = sub(input, x_mean);
              auto x_mean_sub_pow = mul(x_mean_sub, x_mean_sub);
              auto var_sum = sum(x_mean_sub_pow, inner_reduction_axes);
              auto var_sum_bcast = broadcast(var_sum, inner_broadcast_mask);
              auto var = div(var_sum_bcast, num_features);
              auto var_eps = add(var, eps_ptr);
              auto rvar = unaryOp(UnaryOpType::Rsqrt, var_eps);
              auto output = mul(x_mean_sub, rvar);

              // Optional: norm * weight
              if (weight) {
                auto weight_broadcast = broadcast(weight, outer_broadcast_mask);
                output = mul(output, weight_broadcast);
              }

              // Optional: norm * weight + bias
              if (bias) {
                auto bias_broadcast = broadcast(bias, outer_broadcast_mask);
                output = add(output, bias_broadcast);
              }
              if (node->kind() ==
                  c10::Symbol::fromQualString("aten::native_layer_norm")) {
                value_map.emplace(node->output(0)->unique(), output);
                value_map.emplace(node->output(1)->unique(), x_mean);
                value_map.emplace(node->output(2)->unique(), rvar);
              } else if (
                  node->kind() ==
                  c10::Symbol::fromQualString("aten::layer_norm")) {
                value_map.emplace(node->output()->unique(), output);
              }
            },
            // TODO: #ProfileIValue List should update this
            [](const Node* node) -> bool { return true; },
            [](const Node* node) -> OperatorType {
              return OperatorType::Normalization;
            });
      }
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::native_layer_norm_backward(Tensor grad_out, Tensor input, int[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            auto grad_out =
                value_map[node->input(0)->unique()]->as<TensorView>();

            auto input = value_map[node->input(1)->unique()]->as<TensorView>();

            auto norm_shape = constant_as<c10::List<int64_t>>(node->input(2));
            TORCH_INTERNAL_ASSERT(
                norm_shape.has_value(),
                "The Normalized_Shape list is required.");

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
            auto out_mask_list = constant_as<c10::List<bool>>(node->input(7));
            TORCH_INTERNAL_ASSERT(
                out_mask_list.has_value(),
                "output mask for layer_norm_backward");
            std::vector<int> output_mask;
            for (const auto value : out_mask_list->vec()) {
              output_mask.emplace_back(static_cast<int>(value));
            }

            const size_t kNormShapeNumDims = norm_shape->vec().size();
            const size_t kOuterNumDims = input->nDims() - kNormShapeNumDims;

            std::vector<int> outer_reduction_axes(kOuterNumDims);
            std::vector<bool> outer_broadcast_mask(input->nDims(), false);
            for (size_t idx = 0; idx < kOuterNumDims; ++idx) {
              outer_reduction_axes[idx] = idx;
              outer_broadcast_mask[idx] = true;
            }

            std::vector<int> inner_reduction_axes(kNormShapeNumDims);
            std::vector<bool> inner_broadcast_mask(input->nDims(), false);
            Val* num_features = new Double(1);
            for (size_t idx = 0; idx < kNormShapeNumDims; ++idx) {
              const size_t axis = input->nDims() - 1 - idx;
              inner_reduction_axes[idx] = axis;
              inner_broadcast_mask[axis] = true;
              num_features =
                  mul(num_features, input->domain()->domain()[axis]->extent());
            }

            auto x_hat = mul(sub(input, mean), rstd);

            TensorView* grad_x_hat = nullptr;
            if (weight != nullptr) {
              auto* bcast_weight = broadcast(weight, outer_broadcast_mask);
              grad_x_hat = mul(grad_out, bcast_weight);
            } else {
              grad_x_hat = grad_out;
            }

            auto* a = mul(num_features, grad_x_hat);

            auto* b = sum(grad_x_hat, inner_reduction_axes);
            auto* bcast_b = broadcast(b, inner_broadcast_mask);

            auto* c1 = mul(grad_x_hat, x_hat);
            auto* c2 = sum(c1, inner_reduction_axes);
            auto* bcast_c2 = broadcast(c2, inner_broadcast_mask);
            auto* c3 = mul(x_hat, bcast_c2);

            auto* inner = sub(sub(a, bcast_b), c3);

            auto reciprocal_size =
                unaryOp(UnaryOpType::Reciprocal, num_features);
            auto* grad_in = mul(mul(reciprocal_size, rstd), inner);

            if (output_mask[0]) {
              value_map.emplace(node->output(0)->unique(), grad_in);
            } else {
              value_map.emplace(
                  node->output(0)->unique(), TensorViewBuilder().build());
            }

            if (output_mask[1] && weight != nullptr) {
              auto grad_weight =
                  sum(mul(grad_out, x_hat), outer_reduction_axes);
              value_map.emplace(node->output(1)->unique(), grad_weight);
            } else {
              value_map.emplace(
                  node->output(1)->unique(), TensorViewBuilder().build());
            }

            if (output_mask[2] && bias != nullptr) {
              auto grad_bias = sum(grad_out, outer_reduction_axes);
              value_map.emplace(node->output(2)->unique(), grad_bias);
            } else {
              value_map.emplace(
                  node->output(2)->unique(), TensorViewBuilder().build());
            }
          },
          // TODO: #ProfileIValue List should update this
          [](const Node* node) -> bool { return true; },
          [](const Node* node) -> OperatorType {
            return OperatorType::Normalization;
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::softmax.int(Tensor self, int dim, int? dtype) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            auto input = value_map[node->input(0)->unique()]->as<TensorView>();

            auto dim_value = constant_as<int>(node->input(1));
            TORCH_INTERNAL_ASSERT(
                dim_value.has_value(), "dim in softmax is not valid");

            const int kNumberOfDims = input->nDims();
            int kReductionAxis = dim_value.value();
            if (kReductionAxis < 0) {
              kReductionAxis += int(input->nDims());
            }

            std::vector<bool> broadcast_mask(kNumberOfDims, false);
            broadcast_mask[kReductionAxis] = true;

            auto* max_val = max(input, {kReductionAxis});
            auto* bcast_max = broadcast(max_val, broadcast_mask);
            auto* x_max_sub = sub(input, bcast_max);
            auto* exp = unaryOp(UnaryOpType::Exp, x_max_sub);
            auto* sum_exp = sum(exp, {kReductionAxis});
            auto* bcast_sum = broadcast(sum_exp, broadcast_mask);
            auto* output = div(exp, bcast_sum);
            value_map.emplace(node->output()->unique(), output);
          },
          [](const Node* node) -> bool {
            if (node->inputs()[1]->node()->kind() != prim::Constant) {
              return false;
            }
            if (!node->inputs()[2]->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              return false;
            }
            return true;
          },
          [](const Node* node) -> OperatorType {
            return OperatorType::Normalization;
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            auto grad_output =
                value_map[node->input(0)->unique()]->as<TensorView>();

            auto output = value_map[node->input(1)->unique()]->as<TensorView>();

            auto dim_value = constant_as<int>(node->input(2));
            TORCH_INTERNAL_ASSERT(
                dim_value.has_value(), "dim in softmax is not valid");

            auto input = value_map[node->input(3)->unique()]->as<TensorView>();

            const int kNumberOfDims = input->nDims();
            int kReductionAxis = dim_value.value();
            if (kReductionAxis < 0) {
              kReductionAxis += int(input->nDims());
            }

            std::vector<bool> broadcast_mask(kNumberOfDims, false);
            broadcast_mask[kReductionAxis] = true;

            auto* new_grad = mul(grad_output, output);
            auto* sum_new_grad = sum(new_grad, {kReductionAxis});
            auto* bcast_sum = broadcast(sum_new_grad, broadcast_mask);
            auto* output_sum_mul = mul(output, bcast_sum);
            auto* grad_input = sub(new_grad, output_sum_mul);

            value_map.emplace(node->output()->unique(), grad_input);
          },
          [](const Node* node) -> bool {
            if (node->inputs()[2]->node()->kind() != prim::Constant) {
              return false;
            }
            return true;
          },
          [](const Node* node) -> OperatorType {
            return OperatorType::Normalization;
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            auto self = value_map[node->input(0)->unique()];
            auto dims_list = constant_as<c10::List<int64_t>>(node->input(1));
            TORCH_INTERNAL_ASSERT(
                dims_list.has_value(),
                "aten::sum cannot be fused with dynamic axes");
            std::vector<int> dims;
            for (const auto dim : dims_list->vec()) {
              dims.emplace_back(static_cast<int>(dim));
            }
            auto keepdim = constant_as<bool>(node->input(2));
            TORCH_INTERNAL_ASSERT(
                keepdim.has_value(),
                "aten::sum cannot be fused with dynamic keepdim");
            auto out = sum(self->as<TensorView>(), dims, keepdim.value());
            value_map.emplace(node->output()->unique(), out);
          },
          [](const Node* node) -> bool {
            // TODO: support cast of output types
            if (!node->inputs()[3]->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              // We can only handle output as half, float, and double;
              if (const auto opt_ivalue = toIValue(node->input(3))) {
                const auto scalar_type = opt_ivalue->toScalarType();
                if (scalar_type == at::ScalarType::Double ||
                    scalar_type == at::ScalarType::Float ||
                    scalar_type == at::ScalarType::Half) {
                  return true;
                }
              }
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

    {
      std::array<const char*, kNumSumToSize> SumToSize = {
          "aten::_grad_sum_to_size(Tensor(a) self, int[]? size) -> Tensor(a)",
          "aten::sum_to_size(Tensor self, int[] size) -> Tensor"};
      for (auto signature : SumToSize) {
        auto ptr_op = getOperatorForLiteral(signature);
        registerParseRule(
            ptr_op,
            [](const Node* node,
               std::unordered_map<size_t, CgValue>& value_map) -> void {
              auto self = value_map[node->input(0)->unique()];
              auto size_to = constant_as<c10::List<int64_t>>(node->input(1));
              TORCH_INTERNAL_ASSERT(
                  size_to.has_value(),
                  "aten::sum cannot be fused with dynamic axes");
              if (!size_to->empty()) {
                auto out = sum_to(self->as<TensorView>(), size_to->vec());
                value_map.emplace(node->output()->unique(), out);
              } else {
                // We are introducing alias here!
                value_map.emplace(node->output()->unique(), self);
              }
            },
            [](const Node* node) -> bool {
              // we don't support dynamic reduction axes;
              if (node->inputs()[1]->node()->kind() != prim::Constant) {
                return false;
              }
              return true;
              // auto size_to = constant_as<c10::List<int64_t>>(node->input(1));
              // return size_to.has_value() && !size_to->empty();
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
      auto ptr_op = getOperatorForLiteral(
          "aten::type_as(Tensor self, Tensor other) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            auto self = value_map[node->inputs()[0]->unique()];

            // TODO: switch to PyTorch dtype as it's closer to truth.
            // For now, reality is that PyTorch IR profiling information could
            // be missing even with profiling executor, due to upstream
            // transformations between profiling runs to fusion pass.
            auto opt_dtype =
                value_map[node->inputs()[1]->unique()]->getDataType();
            TORCH_INTERNAL_ASSERT(opt_dtype.has_value());

            auto out = castOp(opt_dtype.value(), self);
            value_map.emplace(node->output()->unique(), out);
          });
    }

    {
      // We are not fusing `linear` yet, because we can't codegen efficient gemm
      // However, we still need this here, so PE would insert profile node for
      // this node.
      // During fusion pass, We decompose linear into gemm + elementwise.
      auto ptr_op = getOperatorForLiteral(
          "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            // this entry is created so we do profile input tensors;
            TORCH_INTERNAL_ASSERT(false, "not implemented yet");
          },
          [](const Node* node) -> bool {
            // We only profile `linear` layer with bias.
            if (node->input(2)->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              return false;
            }
            return true;
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "prim::add_optional(Tensor(a) input, Tensor? bias) -> Tensor(a)");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            // this entry is created so we do profile input tensors;
            if (node->input(1)->type()->isSubtypeOf(
                    static_cast<c10::TypePtr>(NoneType::get()))) {
              // forwarding the value;
              value_map.emplace(
                  node->output()->unique(),
                  value_map[node->inputs()[0]->unique()]);
            } else {
              auto lhs = value_map[node->inputs()[0]->unique()];
              auto rhs = value_map[node->inputs()[1]->unique()];

              auto out = binaryOp(BinaryOpType::Add, lhs, rhs);
              value_map.emplace(node->output()->unique(), out);
            }
          });
    }

    {
      auto ptr_op = getOperatorForLiteral(
          "aten::gelu_backward(Tensor grad, Tensor self) -> Tensor");
      registerParseRule(
          ptr_op,
          [](const Node* node,
             std::unordered_map<size_t, CgValue>& value_map) -> void {
            auto grad = value_map[node->inputs()[0]->unique()];
            auto self = value_map[node->inputs()[1]->unique()];

            constexpr double kAlpha = M_2_SQRTPI * M_SQRT1_2 * 0.5;
            const double kHalf = 0.5;

            auto cdf_1 = mul(self, new Double(M_SQRT1_2));
            auto cdf_2 = unaryOp(UnaryOpType::Erf, cdf_1);
            auto cdf_3 = add(cdf_2, new Double(1.));
            auto cdf_4 = mul(cdf_3, new Double(kHalf));

            auto pdf_1 = mul(self, self);
            auto pdf_2 = mul(pdf_1, new Double(-kHalf));
            auto pdf_3 = unaryOp(UnaryOpType::Exp, pdf_2);

            auto out_1 = addcmul(cdf_4, self, pdf_3, new Double(kAlpha));
            auto out_2 = mul(out_1, grad);

            value_map.emplace(node->output()->unique(), out_2);
          });
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
    return registerTensor(val) || registerScalar(val);
  }

  bool registerScalar(const JitValue* val) {
    if (val->type()->isSubtypeOf(static_cast<c10::TypePtr>(FloatType::get()))) {
      CgValue cg_val;
      if (auto ival = constant_as<double>(val)) {
        cg_val = new Double(ival.value());
      } else {
        cg_val = new Double();
      }
      value_map_.emplace(val->unique(), cg_val);
      return true;
    } else if (val->type()->isSubtypeOf(
                   static_cast<c10::TypePtr>(IntType::get()))) {
      CgValue cg_val;
      if (auto ival = constant_as<int64_t>(val)) {
        cg_val = new Int(ival.value());
      } else {
        cg_val = new Int();
      }
      value_map_.emplace(val->unique(), cg_val);
      return true;
    } else if (val->type()->isSubtypeOf(
                   static_cast<c10::TypePtr>(BoolType::get()))) {
      CgValue cg_val;
      if (auto ival = constant_as<bool>(val)) {
        cg_val = new Bool(ival.value());
      } else {
        cg_val = new Bool();
      }
      value_map_.emplace(val->unique(), cg_val);
      return true;
    } else if (val->type()->isSubtypeOf(
                   static_cast<c10::TypePtr>(NoneType::get()))) {
      // TODO: should we consider adding support for NoneType;
      return true;
    } else if (val->type()->cast<ListType>()) {
      // TODO: we don't support list type in codegen yet;
      // This is a WAR to allow axes of reduction to be passed as constant list;
      // We simply ignore conversion if the scalar value is a constant;
      return toIValue(val).has_value();
    }
    return false;
  }

  bool registerTensor(const JitValue* val) {
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

      // TODO: make this a static function in Tensor class;
      // create tensor;
      cg_val = new TensorView(tensor_type);
      value_map_.emplace(val->unique(), cg_val);
      return true;
    }
    return false;
  }

  std::shared_ptr<Graph> graph_;

  // maps from JitValue::unique() to fusion Val;
  std::unordered_map<size_t, CgValue> value_map_;
  // parsing rule registry.
  static std::unordered_map<std::string, RegistrationEntry>
      jit_operator_registry_; // NOLINT

  // pointing cached entry stored in `jit_operator_registry_`
  static std::unordered_map<const FunctionSchema*, const RegistrationEntry*>
      cached_registry_lookup_; // NOLINT

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool init_registry_;
};

std::unordered_map<std::string, IrParser::RegistrationEntry>
    IrParser::jit_operator_registry_; // NOLINT
std::unordered_map<const FunctionSchema*, const IrParser::RegistrationEntry*>
    IrParser::cached_registry_lookup_; // NOLINT

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool IrParser::init_registry_ = true;

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

void profileSize(ProfilingRecord* pr, Node* node, size_t offset) {
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
          false, "profileSize does not support data type: ", value.tagKind());
    }
    if (!pn->hasAttribute(sizeAttr)) {
      pn->is_(sizeAttr, size_vec);
    } else {
      auto profiled_ints = pn->is(sizeAttr);
      TORCH_INTERNAL_ASSERT(
          profiled_ints.size() == size_vec.size() &&
              std::equal(
                  profiled_ints.begin(), profiled_ints.end(), size_vec.begin()),
          "profiling ivalue doesn't support merge");
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
    if (!pn->hasAttribute(intListAttr)) {
      pn->is_(intListAttr, value.toIntVector());
    } else {
      auto profiled_ints = pn->is(intListAttr);
      auto input_ints = value.toIntList();
      TORCH_INTERNAL_ASSERT(
          profiled_ints.size() == input_ints.size() &&
              std::equal(
                  profiled_ints.begin(),
                  profiled_ints.end(),
                  input_ints.begin()),
          "profiling ivalue doesn't support merge");
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
    if (!pn->hasAttribute(boolAttr)) {
      pn->i_(boolAttr, value.toBool());
    } else {
      auto profiled_bool = pn->i(boolAttr);
      auto input_bool = value.toBool();
      TORCH_INTERNAL_ASSERT(
          input_bool == profiled_bool,
          "profiling ivalue doesn't support merge");
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
    if (!pn->hasAttribute(boolListAttr)) {
      auto list = value.toBoolList();
      std::vector<int64_t> val(list.begin(), list.end());
      pn->is_(boolListAttr, val);
    } else {
      auto profiled_ints = pn->is(boolListAttr);
      auto input_bools = value.toBoolList();
      TORCH_INTERNAL_ASSERT(
          profiled_ints.size() == input_bools.size() &&
              std::equal(
                  input_bools.begin(),
                  input_bools.end(),
                  profiled_ints.begin()),
          "profiling ivalue doesn't support merge");
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

bool insertProfileIValue(ProfilingRecord* pr, Node* node, size_t offset) {
  // is skip constant necessary?
  if (node->input(offset)->node()->kind() == prim::Constant) {
    return false;
  }

  static auto dropout_schema =
      getOperatorForLiteral(
          "aten::dropout(Tensor input, float p, bool train) -> Tensor")
          ->schema();
  if (node->matches(dropout_schema)) {
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

  static auto native_dropout_schema =
      getOperatorForLiteral(
          "aten::native_dropout(Tensor input, float p, float scale, bool train) -> (Tensor, Tensor)")
          ->schema();
  if (node->matches(native_dropout_schema)) {
    switch (offset) {
      // argument 3: Is training?
      case 3:
        profileBool(pr, node, offset);
        break;
      default:
        return false;
    }
    return true;
  }

  static auto reduction_operator_schema =
      getOperatorForLiteral(
          "aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)")
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
        profileSize(pr, node, offset);
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

  static auto native_layer_norm_backward_schema =
      getOperatorForLiteral(
          "aten::native_layer_norm_backward(Tensor grad_out, Tensor input, int[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")
          ->schema();
  if (node->matches(native_layer_norm_backward_schema)) {
    switch (offset) {
      case 2:
        profileIntList(pr, node, offset);
        return true;
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
      case 7:
        profileBoolList(pr, node, offset);
        return true;
    }
  }

  return false;
}

void insertProfileNodesForCUDAFuser_(Block* block, ProfilingRecord* pr) {
  for (const auto& n : block->nodes()) {
    for (size_t offset = 0; offset < n->inputs().size(); offset++) {
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
