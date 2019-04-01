#include <string>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {

struct Graph;

namespace script {

// \brief Parse IR from \p STR constructing the corresponding IR in\ GRAPH.
TORCH_API void parseIR(const std::string& str, torch::jit::Graph* graph);

} // namespace script
} // namespace jit
} // namespace torch
