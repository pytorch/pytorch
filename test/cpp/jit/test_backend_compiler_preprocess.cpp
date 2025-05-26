#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_preprocess.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>

namespace torch {
namespace jit {
namespace {
// For this backend, the actual compilation happens in preprocess function AOT.
// Put here for demonstration of backend
// as a whole piece. It's used when compilation is required. A dummy function
// can be passed when there's no usage of compilation in runtime backend lib.
c10::IValue preprocess(
    const Module& mod,
    const c10::Dict<IValue, IValue>& method_compile_spec,
    const BackendDebugHandleGenerator& generate_debug_handles) {
  // The output of this process would produce a dictionary
  // Key: method name.
  // Val: compiled blob (represented by a string).
  c10::Dict<IValue, IValue> compiled(StringType::get(), StringType::get());

  for (const auto& method : mod.get_methods()) {
    auto graph = toGraphFunction(method.function()).graph()->copy();
    // Must inline the graph for debug info map.
    Inline(*graph);
    // This is here because to test module hierarchy we will have
    // getattr nodes which after inlining dont serve any purpose.
    // Without removing them we will run into compilation errors.
    // So eliminate deadcode just remove those getattr nodes.
    EliminateDeadCode(graph);
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto key = method.name();
    auto node_debug_handles = generate_debug_handles(graph);
    std::stringstream ss;
    for (const auto& node : graph->nodes()) {
      switch (node->kind()) {
        case prim::Constant:
          ss << node->kind().toDisplayString() << "#"
             << toIValue(node->output()).value();
          ss << "<debug_handle>" << node_debug_handles[node];
          break;
        // NOLINTNEXTLINE(bugprone-branch-clone)
        case aten::add:
          ss << node->kind().toQualString();
          ss << "<debug_handle>" << node_debug_handles[node];
          break;
        case aten::sub:
          ss << node->kind().toQualString();
          ss << "<debug_handle>" << node_debug_handles[node];
          break;
        default:
          TORCH_CHECK(
              false,
              "The node of ",
              node->kind().toQualString(),
              " is not supported in this compiler. Source code: ",
              node->sourceRange().str());
          break;
      }
      ss << ",";
    }
    std::string blob = ss.str();
    if (!blob.empty()) {
      blob.pop_back();
    }
    compiled.insert(method.name(), blob);
  }
  return compiled;
}

constexpr auto backend_name = "backend_with_compiler_demo";
static auto pre_reg = backend_preprocess_register(backend_name, preprocess);
} // namespace

} // namespace jit
} // namespace torch
