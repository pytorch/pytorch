#include <string>

namespace torch {
namespace jit {

struct Graph;

namespace script {

// \brief Parse IR from \p STR constructing the corresponding IR in\ GRAPH.
void parseIR(const std::string& str, torch::jit::Graph* graph);

} // namespace script
} // namespace jit
} // namespace torch
