#include <torch/csrc/jit/symbolic_script.h>

namespace torch { namespace jit {
  namespace {
    std::mutex lock;
    const std::unordered_map<std::string, std::string> symbolic_scripts({
      {"aten::mul(Tensor self, Tensor other) -> Tensor",
R"(
def forward(self, other):
    return self * other, (self, other)
def backward(ctx, grad_output):
    # type: (Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tensor]
    self, other = ctx
    return (grad_output * other).sum_to_size(self.size()), (grad_output * self).sum_to_size(other.size())
)"},
      });

    // This map is a workaround to cache compiled gradient_pairs. Ideally this graph
    // should be compiled only once and saved in Operator structure.
    // This should be done along with merging into native_functions.yaml.
    std::unordered_map<const FunctionSchema*, GradientPair> cached_gradient_pairs;
  } // anonymous namespace

  c10::optional<GradientPair> gradientInfoForSchema(const FunctionSchema& schema) {
    std::lock_guard<std::mutex> guard(lock);
    auto cache_it = cached_gradient_pairs.find(&schema);
    if (cache_it != cached_gradient_pairs.end()) {
      return cache_it->second;
    } else {
      auto schema_str = canonicalSchemaString(schema);
      auto sym_script_it = symbolic_scripts.find(schema_str);

      if (sym_script_it != symbolic_scripts.end()) {
        // Compile the python code to a script module
        auto cu = std::make_shared<script::Module>();
        script::defineMethodsInModule(cu, symbolic_scripts.at(schema_str), script::nativeResolver, nullptr);
        auto fw_graph = cu->find_method("forward")->graph();
        auto bw_graph = cu->find_method("backward")->graph();

        GradientPair compiled_graphs{fw_graph, bw_graph};
        cached_gradient_pairs.emplace_hint(cache_it, &schema, compiled_graphs);
        return compiled_graphs;
      }
    }
    return c10::nullopt;
  }

  bool hasGradientInfoForSchema(const FunctionSchema& schema) {
    std::lock_guard<std::mutex> guard(lock);
    auto cache_it = cached_gradient_pairs.find(&schema);
    if (cache_it == cached_gradient_pairs.end()) {
      auto schema_str = canonicalSchemaString(schema);
      auto sym_script_it = symbolic_scripts.find(schema_str);
      return !(sym_script_it == symbolic_scripts.end());
    }
    return true;
  }
}}

