#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/passes/decompose_ops.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch {
namespace jit {

Value* decomposeOp(
    Node* op,
    const char* source,
    const std::string& method_name,
    const at::ArrayRef<Value*> inputs) {
  std::shared_ptr<Graph> d_graph;
  std::once_flag flag;
  std::call_once(
      flag,
      [](std::shared_ptr<Graph>* graph_ptr,
         const char* source,
         const std::string& method_name) {
        script::CompilationUnit cu;
        cu.define(source, script::nativeResolver(), nullptr);
        *graph_ptr = cu.get_function(method_name).graph();
      },
      &d_graph,
      source,
      method_name);

  WithInsertPoint insert_guard{op};
  return inlineCallTo(*op->owningGraph(), *d_graph, inputs).at(0);
}

// Yes, no, or no value if we can't tell
c10::optional<bool> isDefined(Value* tensor) {
  if (tensor->type()->isSubtypeOf(TensorType::get())) {
    return true;
  }
  if (tensor->node()->mustBeNone()) {
    return false;
  }
  return {};
}

RegisterOperators reg_bn_unsqueeze({Operator(
    "aten::_ncf_unsqueeze(Tensor self, int ndim) -> Tensor",
    [](const Node* node) {
      return [](Stack& stack) {
        const int64_t ndim = pop(stack).toInt();
        auto self = pop(stack).toTensor();
        c10::SmallVector<int64_t, 8> sizes(ndim, 1);
        AT_ASSERT(self.dim() == 1);
        sizes.at(1) = self.size(0);
        push(stack, self.reshape(sizes));
        return 0;
      };
    })});

RegisterOperators reg_ln_view({Operator(
    "aten::_ncf_view(Tensor self, int[] input_shape, int normalized_ndim) -> Tensor",
    [](const Node* node) {
      return [](Stack& stack) {
        const int64_t normalized_ndim = pop(stack).toInt();
        auto input_shape = pop(stack).toIntListRef();
        auto self = pop(stack).toTensor();
        const int64_t input_ndim = input_shape.size();
        c10::SmallVector<int64_t, 8> sizes(input_ndim, 1);
        for (int i = 0; i < input_ndim - normalized_ndim; ++i) {
          sizes.at(i) = input_shape[i];
        }
        push(stack, self.reshape(sizes));
        return 0;
      };
    })});


static void DecomposeOps(Block* block) {
  static const char* linear_source = R"SCRIPT(
        def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor]):
            output = input.matmul(weight.t())
            if bias is not None:
                output += bias
            return output
      )SCRIPT";
  static const char* addmm_source = R"SCRIPT(
      def addmm(self: Tensor, mat1: Tensor, mat2: Tensor, beta: number = 1.0, alpha: number = 1.0):
          return self + mat1.mm(mat2)
    )SCRIPT";
  static const char* bm_source = R"SCRIPT(
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
      )SCRIPT";
  static const char* lm_source = R"SCRIPT(
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
      )SCRIPT";

  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks()) {
      DecomposeOps(sub);
    }
    if (it->matches("aten::linear(Tensor input, Tensor weight, Tensor? bias) -> Tensor")) {
      Value* input = it->namedInput(attr::input);
      Value* weight = it->namedInput(attr::weight);
      Value* bias = it->namedInput(attr::bias);

      WithInsertPoint guard(*it);

      Graph* graph = it->owningGraph();
      int ndim = input->type()->cast<DimensionedTensorType>()->dim();
      Value* new_output = nullptr;
      if (ndim == 2 && isDefined(bias).value_or(false)) {
        // if ndim == 2 and bias is statically defined, dispatch to addmm decomposition
        Value* transposed_weight = graph->insert(aten::t, {weight});
        Value* one = graph->insertConstant(1);
        std::vector<Value*> inputs{bias, input, transposed_weight, one, one};
        new_output = decomposeOp(*it, addmm_source, "addmm", inputs);
      } else {
        // otherwise dispatch to normal linear decomposition
        new_output = decomposeOp(*it, linear_source, "linear", it->inputs());
      }
      new_output->setType(it->output()->type());
      it->output()->replaceAllUsesWith(new_output);
      it.destroyCurrent();
    } else if (it->matches(
            "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor",
            /*const_inputs=*/{attr::beta, attr::alpha})) {
      // For the case where we have an addmm where alpha and beta are Attributes
      // and both of those scalars are equal to 1.0, decompose this into an mm
      // followed by an add so that it can go through the existing optimization (batchmm)
      if (it->get<at::Scalar>(attr::alpha)->toDouble() != 1.0 ||
          it->get<at::Scalar>(attr::beta)->toDouble() != 1.0) {
        continue;
      }

      WithInsertPoint guard(*it);

      Value* new_output = decomposeOp(*it, addmm_source, "addmm", it->inputs());
      // Set the output of the decomposed graph to have the same output type as the
      // original op otherwise the canonicalized graph will have
      // TensorType as the output of this node which is incorrect
      new_output->setType(it->output()->type());
      it->output()->replaceAllUsesWith(new_output);
      it.destroyCurrent();
    } else if (it->matches(
          "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor")) {
      Value* weight = it->namedInput(attr::weight);
      Value* bias = it->namedInput(attr::bias);
      if (!isDefined(weight).has_value() || !isDefined(bias).has_value()) {
        // If we can't determine if weight and bias is defined statically there's
        // really no point in decomposing normalization into simpler ops, since it
        // won't get fused into a single kernel later during the fusion
        continue;
      }
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

      Value* new_output = decomposeOp(*it, bm_source, "batch_norm", inputs);
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
    } else if (it->matches(
          "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps, bool cudnn_enable) -> Tensor")) {
      Value* weight = it->namedInput(attr::weight);
      Value* bias = it->namedInput(attr::bias);
      if (!isDefined(weight).has_value() || !isDefined(bias).has_value()) {
        // If we can't determine if weight and bias is defined statically there's
        // really no point in decomposing normalization into simpler ops, since it
        // won't get fused into a single kernel later during the fusion
        continue;
      }
      WithInsertPoint insert_guard{*it};
      Graph* graph = it->owningGraph();

      std::vector<Value*> inputs{
          it->namedInput(attr::input),
          it->namedInput(attr::normalized_shape),
          it->namedInput(attr::eps),
          it->namedInput(attr::cudnn_enable)};

      Value* new_output = decomposeOp(*it, lm_source, "layer_norm", inputs);
      auto weight_defined = isDefined(weight).value();
      auto bias_defined = isDefined(bias).value();
      if (weight_defined && bias_defined) {
        new_output = graph->insert(aten::addcmul, {bias, new_output, weight});
      } else if (weight_defined) {
        new_output = graph->insert(aten::mul, {new_output, weight});
      } else if (bias_defined) {
        new_output = graph->insert(aten::add, {new_output, bias});
      }
      it->output()->replaceAllUsesWith(new_output);
      it.destroyCurrent();
    }

  }
}

void DecomposeOps(const std::shared_ptr<Graph>& graph) {
  DecomposeOps(graph->block());
  EliminateDeadCode(graph);
}

} // namespace jit
} // namespace torch
