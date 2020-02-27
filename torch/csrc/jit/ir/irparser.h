
copy: fbcode/caffe2/torch/csrc/jit/ir/irparser.h
copyrev: 9cded40c7c296e5935090c22060cc1701a5da5ab

#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <string>
#include <unordered_map>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {

struct Graph;
struct Value;

namespace script {

// \brief Parse IR from \p STR constructing the corresponding IR in\ GRAPH.
TORCH_API void parseIR(const std::string& str, torch::jit::Graph* graph);

/** \brief Parse IR from \p STR constructing the corresponding IR in\ GRAPH.
 *
 * \p VMAP is filled with String to Value pairs allowing to index Values in the
 * newly created graph by their name in the original IR string.
 */
TORCH_API void parseIR(
    const std::string& str,
    torch::jit::Graph* graph,
    std::unordered_map<std::string, Value*>& vmap);

} // namespace script
} // namespace jit
} // namespace torch
