#include <torch/csrc/jit/symbolic_script.h>

namespace torch {
namespace jit {
namespace {
std::mutex lock;
const std::vector<std::string> functions = {
    R"(
        def mul(self, other):
            def backward(grad_output):
                grad_self = (grad_output * other).sum_to_size(self.size())
                grad_other = (grad_output * self).sum_to_size(other.size())
                return grad_self, grad_other
            return self * other, backward

        def adaptive_avg_pool2d(self,
                                output_size: List[int]):
            def backward(grad_output):
                grad_self = torch.adaptive_avg_pool2d_backward(grad_output, self)
                return grad_self, None

            return torch.adaptive_avg_pool2d(self, output_size), backward
      )"};
std::unordered_map<std::string, GradientPair> schema_to_graphs;

// This map is a workaround to cache compiled gradient_pairs. Ideally this graph
// should be compiled only once and saved in Operator structure.
// This should be done along with merging into native_functions.yaml.
std::unordered_map<const FunctionSchema*, GradientPair> cached_gradient_pairs;
} // anonymous namespace

std::pair<std::shared_ptr<Graph>, Value*> extractClosure(Value* closure) {
  AT_CHECK(
      closure->node()->kind() == prim::TupleConstruct,
      "closure must be a literal tuple construct");
  Value* fn = closure->node()->inputs().at(0);
  Value* context = closure->node()->inputs().at(1);

  AT_CHECK(
      fn->node()->kind() == prim::Function,
      "closure tuple must contain a prim::Function");
  return std::make_pair(fn->node()->g(attr::Subgraph), context);
}

Argument originalReturnType(const TupleTypePtr& tup) {
  AT_CHECK(tup->elements().size() > 1);
  if (tup->elements().size() == 2)
    return Argument("", tup->elements().at(0));
  std::vector<TypePtr> types = tup->elements().vec();
  types.pop_back();
  return Argument("", TupleType::create(std::move(types)));
}

void loadModule(const std::shared_ptr<script::Module>& module) {
  for (const auto& method_ : module->get_methods()) {
    const auto& method = method_.value();
    GradientPair pair;
    pair.forward = method->graph();

    // lookup the backward function
    Node* forward_tuple = pair.forward->outputs().at(0)->node();

    if (forward_tuple->kind() != prim::TupleConstruct) {
      throw script::ErrorReport(forward_tuple->getSourceLocation())
          << "gradient must return literal a tuple";
    }

    Value* context;
    std::tie(pair.backward, context) =
        extractClosure(forward_tuple->inputs().back());

    // do surgery on the forward function to remove the closure tuple and
    // replace it with the context variable:
    //  backward = (<lambda>, context_tuple)
    //  return original, backward
    //  -----
    //  return original, context_tuple
    std::vector<Value*> new_inputs = forward_tuple->inputs().vec();
    new_inputs.back() = context;
    Value* new_tuple =
        pair.forward->appendNode(pair.forward->createTuple(new_inputs))
            ->output();
    pair.forward->eraseOutput(0);
    pair.forward->registerOutput(new_tuple);
    forward_tuple->destroy();

    // derive schema from original function's schema:
    const FunctionSchema& loaded_schema = method->getSchema();
    FunctionSchema actual_schema(
        Symbol::aten(loaded_schema.name()),
        loaded_schema.arguments(),
        {originalReturnType(new_tuple->type()->expect<TupleType>())});
    std::string key = canonicalSchemaString(actual_schema);
    schema_to_graphs[key] = std::move(pair);
  }
}

void loadFunctions() {
  for (const std::string& str : functions) {
    auto cu = std::make_shared<script::Module>();
    script::defineMethodsInModule(cu, str, script::nativeResolver, nullptr);
    loadModule(cu);
  }
}

c10::optional<GradientPair> gradientInfoForSchema(
    const FunctionSchema& schema) {
  std::lock_guard<std::mutex> guard(lock);
  if (schema_to_graphs.size() == 0) {
    loadFunctions();
  }
  auto cache_it = cached_gradient_pairs.find(&schema);
  if (cache_it != cached_gradient_pairs.end()) {
    return cache_it->second;
  } else {
    auto schema_str = canonicalSchemaString(schema);
    auto sym_script_it = schema_to_graphs.find(schema_str);
    if (sym_script_it != schema_to_graphs.end()) {
      cached_gradient_pairs.emplace_hint(
          cache_it, &schema, sym_script_it->second);
      return sym_script_it->second;
    }
  }
  return c10::nullopt;
}

bool hasGradientInfoForSchema(const FunctionSchema& schema) {
  return gradientInfoForSchema(schema).has_value();
}

} // namespace jit
} // namespace torch
