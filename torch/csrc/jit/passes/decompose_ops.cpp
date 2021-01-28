#include <torch/csrc/jit/passes/decompose_ops.h>

#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch {
namespace jit {

namespace {
c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}
} // namespace

// helper to determine if an optional tensor argument/value passed in is
// statically defined (neither a None constant nor a Optional[Tensor] type)
// return yes, no, or no value if we can't tell
c10::optional<bool> isDefined(Value* tensor) {
  if (tensor->type()->isSubtypeOf(TensorType::get())) {
    return true;
  }
  if (tensor->node()->mustBeNone()) {
    return false;
  }
  return {};
}

bool isDecomposableNorm(Node* normalize_op) {
  static const OperatorSet decomposable_normalization_ops = {
      "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
      "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps, bool cudnn_enable) -> Tensor",
  };
  Value* input = normalize_op->namedInput(attr::input);
  if (!input->type()->isSubtypeOf(TensorType::get())) {
    return false;
  }
  auto device = input->type()->expectRef<TensorType>().device();
  // As of now, we do the decomposition for batchnorm/layernorm on GPU device
  // only
  if (!device || (*device).is_cpu()) {
    return false;
  }

  if (normalize_op->isMemberOf(decomposable_normalization_ops)) {
    // If we can't determine if weight and bias is defined statically there's
    // really no point in decomposing normalization into simpler ops, since it
    // won't get fused into a single kernel.
    return isDefined(normalize_op->namedInput(attr::weight)).has_value() &&
        isDefined(normalize_op->namedInput(attr::bias)).has_value();
  }
  return false;
}

RegisterOperators reg_ops(
    {Operator(
         "aten::_ncf_unsqueeze(Tensor(a) self, int ndim) -> Tensor(a)",
         [](Stack* stack) {
           const int64_t ndim = pop(stack).toInt();
           auto self = pop(stack).toTensor();
           c10::SmallVector<int64_t, 8> sizes(ndim, 1);
           AT_ASSERT(self.dim() == 1);
           sizes.at(1) = self.size(0);
           push(stack, self.reshape(sizes));
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::_ncf_view(Tensor(a) self, int[] input_shape, int normalized_ndim) -> Tensor(a)",
         [](Stack* stack) {
           const int64_t normalized_ndim = pop(stack).toInt();
           auto input_shape = pop(stack).toIntList();
           auto self = pop(stack).toTensor();
           const int64_t input_ndim = input_shape.size();
           c10::SmallVector<int64_t, 8> sizes(input_ndim, 1);
           for (int i = 0; i < input_ndim - normalized_ndim; ++i) {
             sizes.at(i) = input_shape.get(i);
           }
           push(stack, self.reshape(sizes));
         },
         aliasAnalysisFromSchema())});

bool DecomposeOps(Block* block, CompilationUnit& decompose_funcs) {
  bool decomposed = false;
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks()) {
      DecomposeOps(sub, decompose_funcs);
    }

    if (it->matches(
            "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor",
            /*const_inputs=*/{attr::beta, attr::alpha})) {
      // For the case where we have an addmm where alpha and beta are Attributes
      // and both of those scalars are equal to 1.0, decompose this into an mm
      // followed by an add so that it can go through the existing optimization
      // (batchmm)
      if (it->get<at::Scalar>(attr::alpha)->toDouble() != 1.0 ||
          it->get<at::Scalar>(attr::beta)->toDouble() != 1.0) {
        continue;
      }

      decomposed = true;
      WithInsertPoint guard(*it);
      std::shared_ptr<Graph> d_graph =
          decompose_funcs.get_function("addmm").graph();
      Value* new_output =
          insertGraph(*it->owningGraph(), *d_graph, it->inputs()).at(0);
      // Set the output of the decomposed graph to have the same output type as
      // the original op otherwise the canonicalized graph will have TensorType
      // as the output of this node which is incorrect
      new_output->setType(it->output()->type());
      it->output()->replaceAllUsesWith(new_output);
      it.destroyCurrent();
    } else if (
        it->matches(
            "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor")) {
      if (!isDecomposableNorm(*it)) {
        continue;
      }
      decomposed = true;
      WithInsertPoint insert_guard{*it};
      Graph* graph = it->owningGraph();
      Value* input = it->namedInput(attr::input);
      Value* input_dim = graph->insert(aten::dim, {input});
      std::vector<Value*> inputs{
          input,
          it->namedInput(attr::running_mean),
          it->namedInput(attr::running_var),
          it->namedInput(attr::training),
          it->namedInput(attr::momentum),
          it->namedInput(attr::eps)};

      // inline the compiled decomposed batchnorm
      std::shared_ptr<Graph> d_graph =
          decompose_funcs.get_function("batch_norm").graph();
      Value* new_output = insertGraph(*graph, *d_graph, inputs).at(0);

      // post processing the graph
      Value* weight = it->namedInput(attr::weight);
      Value* bias = it->namedInput(attr::bias);
      if (isDefined(weight).value()) {
        Value* expanded_weight =
            graph->insert(aten::_ncf_unsqueeze, {weight, input_dim});
        new_output = graph->insert(aten::mul, {new_output, expanded_weight});
      }
      if (isDefined(bias).value()) {
        Value* expanded_bias =
            graph->insert(aten::_ncf_unsqueeze, {bias, input_dim});
        new_output = graph->insert(aten::add, {new_output, expanded_bias});
      }
      it->output()->replaceAllUsesWith(new_output);
      it.destroyCurrent();
    } else if (
        it->matches(
            "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps, bool cudnn_enable) -> Tensor")) {
      if (!isDecomposableNorm(*it)) {
        continue;
      }
      decomposed = true;
      WithInsertPoint insert_guard{*it};
      Graph* graph = it->owningGraph();
      std::vector<Value*> inputs{
          it->namedInput(attr::input),
          it->namedInput(attr::normalized_shape),
          it->namedInput(attr::eps),
          it->namedInput(attr::cudnn_enable)};

      // inline the compiled decomposed layernorm
      std::shared_ptr<Graph> d_graph =
          decompose_funcs.get_function("layer_norm").graph();
      Value* new_output = insertGraph(*graph, *d_graph, inputs).at(0);

      // post processing the graph
      Value* weight = it->namedInput(attr::weight);
      Value* bias = it->namedInput(attr::bias);
      if (isDefined(weight).value()) {
        new_output = graph->insert(aten::mul, {new_output, weight});
      }
      if (isDefined(bias).value()) {
        new_output = graph->insert(aten::add, {new_output, bias});
      }
      it->output()->replaceAllUsesWith(new_output);
      it.destroyCurrent();
    }
  }
  return decomposed;
}

void DecomposeOps(std::shared_ptr<Graph>& graph) {
  static CompilationUnit decompose_funcs(R"SCRIPT(
      def addmm(self: Tensor, mat1: Tensor, mat2: Tensor, beta: number = 1.0, alpha: number = 1.0):
          return self + mat1.mm(mat2)

      def batch_norm(input : Tensor, running_mean : Optional[Tensor], running_var : Optional[Tensor], training : bool, momentum : float, eps : float) -> Tensor:
          if training:
              norm_mean, norm_var = torch.batch_norm_update_stats(input, running_mean, running_var, momentum)
          else:
              norm_mean = torch._unwrap_optional(running_mean)
              norm_var = torch._unwrap_optional(running_var)
          norm_mean = torch._ncf_unsqueeze(norm_mean, input.dim())
          norm_var = torch._ncf_unsqueeze(norm_var, input.dim())
          norm_invstd = 1 / (torch.sqrt(norm_var + eps))
          return ((input - norm_mean) * norm_invstd)

      def layer_norm(input : Tensor, normalized_shape : List[int], eps : float, cudnn_enable : bool) -> Tensor:
          input_ndim = input.dim()
          normalized_ndim = len(normalized_shape)
          n = 1
          for i in range(input_ndim - normalized_ndim):
              n *= input.size(i)
          input_reshape = input.contiguous().view(1, n, -1)
          mean, invstd = torch.batch_norm_stats(input_reshape, eps)
          input_shape = input.size()
          mean = torch._ncf_view(mean, input_shape, normalized_ndim)
          invstd = torch._ncf_view(invstd, input_shape, normalized_ndim)

          return (input - mean) * invstd
      )SCRIPT");
  bool is_decomposed = DecomposeOps(graph->block(), decompose_funcs);
  if (is_decomposed) {
    // we only re-run those passes when the graph get decomposed
    PropagateInputShapes(graph);
    ConstantPropagation(graph);
    EliminateDeadCode(graph);
  }
}

} // namespace jit
} // namespace torch
