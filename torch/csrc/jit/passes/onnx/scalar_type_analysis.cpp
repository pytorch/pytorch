#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/scalar_type_analysis.h>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

namespace {
const int ONNX_OPSET_14 = 14;

static const std::unordered_map<c10::ScalarType, int, ScalarTypeHashFunction>
    scalarTypeToONNXTypeMap = {
        {c10::kFloat, 1},
        {c10::kByte, 2},
        {c10::kChar, 3},
        {c10::kShort, 5},
        {c10::kInt, 6},
        {c10::kLong, 7},
        {c10::kBool, 9},
        {c10::kHalf, 10},
        {c10::kDouble, 11},
        {c10::kQInt8, 12},
        {c10::kQUInt8, 13},
        {c10::kQInt32, 14},
        {c10::kBFloat16, 15},
};

static int64_t ScalarTypeToONNXType(const c10::ScalarType& st) {
  int64_t onnx_type = -1;
  const auto it = scalarTypeToONNXTypeMap.find(st);
  if (it != scalarTypeToONNXTypeMap.end()) {
    onnx_type = it->second;
  }
  return onnx_type;
}

// For these operators, all inputs and outputs share the same scalar type.
// There is no operator-wise special case handling needed.
static const std::unordered_set<NodeKind> standardOps = {
    onnx::Add,
    onnx::Concat,
    onnx::Div,
    onnx::Gemm,
    onnx::Min,
    onnx::Max,
    onnx::Mod,
    onnx::Mul,
    onnx::Pow,
    onnx::Sub,
    onnx::MatMul,
};

// For these operators, all inputs share the same scalar type.
// The output scalar type is always Bool.
static const std::unordered_set<NodeKind> comparisonOps = {
    onnx::Greater,
    onnx::Less,
    onnx::Equal,
    onnx::GreaterOrEqual,
    onnx::LessOrEqual,
};

static bool IsStandardOp(const NodeKind& nkind) {
  return standardOps.find(nkind) != standardOps.end();
}

static bool IsComparisonOp(const NodeKind& nkind) {
  return comparisonOps.find(nkind) != comparisonOps.end();
}

static TensorTypePtr CreateProfiledTensorTypeWithScalarType(
    const TensorTypePtr& typePtr,
    const c10::ScalarType& scalar_type) {
  TORCH_INTERNAL_ASSERT(typePtr != nullptr);
  return typePtr->withScalarType({scalar_type});
}

static bool IsImplicitCastSupported(const NodeKind& nodeKind) {
  return IsStandardOp(nodeKind) || IsComparisonOp(nodeKind);
}

static c10::optional<c10::ScalarType> PromoteScalarTypes(
    const std::vector<c10::ScalarType>& types) {
  if (types.empty()) {
    return c10::nullopt;
  }
  auto st = types[0];
  for (const auto i : c10::irange(1, types.size())) {
    st = c10::promoteTypes(st, types[i]);
  }
  return st;
}

// Type promotion between scalars and tensors
// per logic here
// https://pytorch.org/docs/master/tensor_attributes.html#tensor-attributes
static c10::optional<c10::ScalarType> PromoteScalarTypesWithCategory(
    const std::vector<c10::ScalarType>& typesFromTensors,
    const std::vector<c10::ScalarType>& typesFromScalars) {
  auto typeFromTensor = PromoteScalarTypes(typesFromTensors);
  auto typeFromScalar = PromoteScalarTypes(typesFromScalars);

  auto getTypeCategory = [](c10::ScalarType t) {
    if (c10::kBool == t) {
      return 1;
    }
    if (c10::isIntegralType(t, /*includeBool=*/false)) {
      return 2;
    }
    if (c10::isFloatingType(t)) {
      return 3;
    }
    return 0;
  };

  if (c10::nullopt == typeFromScalar) {
    return typeFromTensor;
  } else if (c10::nullopt == typeFromTensor) {
    return typeFromScalar;
  }

  auto typeCategoryFromTensor = getTypeCategory(typeFromTensor.value());
  auto typeCategoryFromScalar = getTypeCategory(typeFromScalar.value());

  if (typeCategoryFromScalar > typeCategoryFromTensor) {
    return typeFromScalar;
  }
  return typeFromTensor;
}

static c10::optional<c10::ScalarType> InferExpectedScalarType(const Node* n) {
  std::vector<c10::ScalarType> typesFromTensors;
  std::vector<c10::ScalarType> typesFromScalars;

  auto get_scalar_type =
      [](const Value* input) -> c10::optional<at::ScalarType> {
    if (auto* tensor_type = input->type()->castRaw<TensorType>()) {
      return tensor_type->scalarType();
    }
    return c10::nullopt;
  };
  auto emplace_type_from_scalar =
      [&typesFromTensors, &typesFromScalars](at::ScalarType scalar_type) {
        // Mimic PyTorch scalar type promotion logic
        // from https://github.com/pytorch/pytorch/issues/9515
        // Quoting:
        //    A Tensor is a considered a "wrapped number" if it is
        //    auto-wrapped from a C++ or Python number type. Integer types are
        //    wrapped as 0-dim int64 tensors and floating-point types are
        //    wrapped as 0-dim double tensors.
        auto default_scalar_type =
            at::typeMetaToScalarType(at::get_default_dtype());
        switch (scalar_type) {
          case at::kDouble:
            // floating-point numbers wrapped as double tensors are
            // considered to have default type, instead of double.
            typesFromScalars.emplace_back(default_scalar_type);
            break;
          case at::kLong:
          case at::kBool:
            // bool and integer numbers remain the same type.
            typesFromScalars.emplace_back(scalar_type);
            break;
          default:
            // other types are not from wrapped numbers,
            // track them as types from tensors.
            typesFromTensors.emplace_back(scalar_type);
            break;
        }
      };

  std::for_each(
      n->inputs().begin(), n->inputs().end(), [&](const Value* input) {
        auto nkind = input->node()->kind();
        if (nkind == onnx::Gather &&
            input->node()->input(0)->node()->kind() == onnx::Shape) {
          // This is a special pattern generated by code like `dim_size =
          // x.size(0)`. It gets converted to the below ONNX IR graph
          //    %1 : Long() = onnx::Constant[value={0}]()
          //    %2 : Tensor = onnx::Shape(%x)
          //    %dim_size : Long() = onnx::Gather(%2, %1)
          // `dim_size` is treated in PyTorch as Scalar.
          // However, in the ONNX IR graph, it is an output of onnx::Gather,
          // which is by default considered as a tensor.
          typesFromScalars.emplace_back(c10::kLong);
        } else if (nkind == onnx::Constant) {
          auto tensor = input->node()->t(attr::value);
          auto rank = tensor.dim();
          auto scalar_type = tensor.scalar_type();

          if (rank == 0) {
            emplace_type_from_scalar(scalar_type);
          } else {
            typesFromTensors.emplace_back(scalar_type);
          }
        } else if (nkind == prim::Param) {
          // ONNX doesn't support scalar as graph input. When
          // seeing a scalar input, we convert its expected type to tensor.
          if (auto scalar_type = get_scalar_type(input)) {
            auto tensor_type = input->type()->castRaw<TensorType>();
            // get_scalar_type returns non-null value already guarantees
            // that the input has a valid tensor_type.
            TORCH_INTERNAL_ASSERT(nullptr != tensor_type);
            auto rank = tensor_type->dim();
            if (rank && rank.value() == 0) {
              emplace_type_from_scalar(scalar_type.value());
            } else {
              typesFromTensors.emplace_back(scalar_type.value());
            }
          }
        } else if (auto scalar_type = get_scalar_type(input)) {
          typesFromTensors.emplace_back(*scalar_type);
        }
      });

  c10::optional<c10::ScalarType> st = c10::nullopt;
  const auto output_st = get_scalar_type(n->output());

  if (IsComparisonOp(n->kind())) {
    // For comparison ops, always promote scalar type to highest among inputs,
    // regardless if that input is a tensor or scalar.
    typesFromScalars.insert(
        typesFromScalars.end(),
        typesFromTensors.begin(),
        typesFromTensors.end());
    st = PromoteScalarTypes(typesFromScalars);
  } else {
    if (output_st) {
      // If output scalar type is available, use that.
      st = output_st;
    } else {
      // PyTorch now does implicit type promotion regardless whether the inputs
      // are tensors or scalars. (Previously only scalars support implicit
      // casting).
      // Per logic here
      // https://pytorch.org/docs/master/tensor_attributes.html#tensor-attributes
      st = PromoteScalarTypesWithCategory(typesFromTensors, typesFromScalars);
    }
  }

  return st;
}

static c10::optional<c10::ScalarType> LowPrecisionCastForStandardOps(
    const Node* n,
    const c10::ScalarType& scalar_type) {
  // Some of standardOps do not support uint8\int8\int16 type for ONNX
  // opset version < 14.
  // Fix in this ONNX PR:
  // https://github.com/onnx/onnx/pull/3334
  if (n->kind() != onnx::Gemm && IsStandardOp(n->kind()) &&
      (scalar_type == c10::kByte || scalar_type == c10::kChar ||
       scalar_type == c10::kShort)) {
    return c10::kLong;
  }
  return scalar_type;
}

static void UpdateScalarTypeForInputs(
    Node* n,
    const c10::ScalarType& scalar_type) {
  const int64_t onnx_type = ScalarTypeToONNXType(scalar_type);
  if (onnx_type < 0) {
    TORCH_WARN(
        "ONNX Scalar Type Analysis - Scalar type: ",
        c10::toString(scalar_type),
        " of input tensor in operator: ",
        n->kind().toDisplayString(),
        " not supported in ONNX. ");
    return;
  }

  for (auto input : n->inputs()) {
    auto input_tensor_type = input->type()->cast<TensorType>();
    auto input_scalar_type =
        input_tensor_type ? input_tensor_type->scalarType() : c10::nullopt;

    if ((input->node()->kind() == onnx::Constant) ||
        (input_scalar_type && (*input_scalar_type != scalar_type))) {
      if (input->node()->kind() == onnx::Constant) {
        // Fix up the scalar directly instead of inserting a cast operator.
        // TODO: Keep only the else branch once constant_folding is enabled by
        // default.
        at::Tensor val = input->node()->t(attr::value);
        at::Tensor new_val = val.to(scalar_type);
        Node* const_node = n->owningGraph()->create(onnx::Constant);
        const_node->t_(attr::value, new_val);
        const_node->insertBefore(n);
        const_node->output()->setType(TensorType::create(new_val));
        const_node->copyMetadata(n);
        n->replaceInputWith(input, const_node->output());
      } else {
        Node* cast_node = n->owningGraph()->create(onnx::Cast);
        cast_node->addInput(input);
        cast_node->i_(attr::to, onnx_type);
        cast_node->insertBefore(n);
        cast_node->output()->setType(CreateProfiledTensorTypeWithScalarType(
            input_tensor_type, scalar_type));
        cast_node->copyMetadata(n);
        n->replaceInputWith(input, cast_node->output());
      }
    }
  }
}

static void UpdateScalarTypeForOutput(
    Node* n,
    const c10::ScalarType& scalar_type) {
  if (auto output_tensor_type = n->output()->type()->cast<TensorType>()) {
    n->output()->setType(CreateProfiledTensorTypeWithScalarType(
        output_tensor_type, scalar_type));
  }
}

static void RecoverScalarTypeForOutput(
    Value* out,
    const c10::ScalarType& scalar_type) {
  Node* n = out->node();
  TORCH_INTERNAL_ASSERT(nullptr != n);
  const int64_t onnx_type = ScalarTypeToONNXType(scalar_type);
  Node* cast_node = n->owningGraph()->create(onnx::Cast, 1);
  cast_node->addInput(out);
  cast_node->i_(attr::to, onnx_type);
  cast_node->insertAfter(n);
  cast_node->copyMetadata(n);
  out->replaceAllUsesAfterNodeWith(cast_node, cast_node->output());
}

// This example error found when exports transfo_xl model using add op in uint8
// type, as below:
// if self.same_length:
//     all_ones = word_emb.new_ones((qlen, klen), dtype=torch.uint8)
//     mask_len = klen - self.mem_len
//     if mask_len > 0:
//         mask_shift_len = qlen - mask_len
//     else:
//         mask_shift_len = qlen
//     dec_attn_mask = (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones,
//     -mask_shift_len))[:, :, None]  # -1
//
// `all_ones is` an uint8 tensor, But the calculation of `dec_attn_mask` using
// add(+) op to get the uint8 result. Reference Link:
// https://github.com/huggingface/transformers/blob/b020a736c374460af1b34267283f957988350630/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L936
static void LowPrecisionCastNodeForStandardOps(Node* n, int opset_version) {
  TORCH_INTERNAL_ASSERT(n->outputs().size() == 1);
  if (n->output()->type()->cast<TensorType>() == nullptr ||
      n->output()->type()->cast<TensorType>()->scalarType() == c10::nullopt) {
    // skip LowPrecisionCast if op output type is null.
    return;
  }
  auto output_scalar_type =
      n->output()->type()->cast<TensorType>()->scalarType().value();
  for (size_t i = 0; i < n->inputs().size(); ++i) {
    if (n->input(i)->type()->cast<TensorType>() == nullptr ||
        n->input(i)->type()->cast<TensorType>()->scalarType() == c10::nullopt) {
      // skip LowPrecisionCast if any op input type node is null.
      return;
    }
    auto input_tensor_type =
        n->input(i)->type()->cast<TensorType>()->scalarType().value();
    TORCH_INTERNAL_ASSERT(output_scalar_type == input_tensor_type);
  }

  // The LowPrecision problem will be fixed in ONNX opset 14.
  if (opset_version < ONNX_OPSET_14) {
    auto expected_scalar_type_cast =
        LowPrecisionCastForStandardOps(n, output_scalar_type);
    UpdateScalarTypeForInputs(n, *expected_scalar_type_cast);
    if (output_scalar_type != *expected_scalar_type_cast) {
      // If input type is changed, convert it to the original type.
      RecoverScalarTypeForOutput(n->output(), output_scalar_type);
    }
  }
}

static void ImplicitCastNodeForONNX(Node* n) {
  if (IsImplicitCastSupported(n->kind())) {
    auto expected_scalar_type = InferExpectedScalarType(n);
    if (expected_scalar_type) {
      UpdateScalarTypeForInputs(n, *expected_scalar_type);
      if (!IsComparisonOp(n->kind())) {
        UpdateScalarTypeForOutput(n, *expected_scalar_type);
      }
    }
  }
}

static void ImplicitCastForONNX(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    for (auto sub : it->blocks()) {
      ImplicitCastForONNX(sub);
    }

    ImplicitCastNodeForONNX(*it);
  }
  EliminateDeadCode(
      block, true, DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
}

static void LowPrecisionCastForStandardOpsONNX(
    Block* block,
    int opset_version) {
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    for (auto sub : it->blocks()) {
      LowPrecisionCastForStandardOpsONNX(sub, opset_version);
    }

    if (IsStandardOp(it->kind())) {
      LowPrecisionCastNodeForStandardOps(*it, opset_version);
    }
  }
  EliminateDeadCode(
      block, true, DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
}
} // anonymous namespace

void ScalarTypeAnalysisForONNX(
    const std::shared_ptr<Graph>& graph,
    bool lowprecision_cast,
    int opset_version) {
  GRAPH_DUMP("Before ScalarTypeAnalysisForONNX: ", graph);
  ImplicitCastForONNX(graph->block());
  if (lowprecision_cast) {
    LowPrecisionCastForStandardOpsONNX(graph->block(), opset_version);
  }
  GRAPH_DUMP("After ScalarTypeAnalysisForONNX: ", graph);
}

void ScalarTypeAnalysisNodeForONNX(Node* n) {
  ImplicitCastNodeForONNX(n);
}

} // namespace jit
} // namespace torch
