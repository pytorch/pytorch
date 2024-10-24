#pragma once

#include <optional>
#include <string>
#include <unordered_map>

#include <torch/csrc/Export.h>

namespace torch::jit {

struct Graph;
struct Value;

// \brief Parse IR from \p STR constructing the corresponding IR in\ GRAPH.
// if parse_tensor_constants is true will construct empty tensors
// for Tensor constants with random or unitialized contents, otherwise will
// throw
TORCH_API void parseIR(
    const std::string& str,
    torch::jit::Graph* graph,
    bool parse_tensor_constants = false);

/** \brief Parse IR from \p STR constructing the corresponding IR in\ GRAPH.
 *
 * \p VMAP is filled with String to Value pairs allowing to index Values in the
 * newly created graph by their name in the original IR string.
 * if parse_tensor_constants is true will construct empty tensors
 * for Tensor constants with random or unitialized contents, otherwise will
 * throw
 */
TORCH_API void parseIR(
    const std::string& str,
    torch::jit::Graph* graph,
    std::unordered_map<std::string, Value*>& vmap,
    bool parse_tensor_constants = false);

} // namespace torch::jit
