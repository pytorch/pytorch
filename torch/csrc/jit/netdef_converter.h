#pragma once
#include <caffe2/proto/caffe2_pb.h>
#include <torch/csrc/jit/ir.h>
#include <unordered_map>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {

/** \brief Convert a caffe2 NetDef to PyTorch IR.
 *
 * The NetDef \p net is converted and the result is stored in the
 * torch::jit::Graph \p graph. The function also records name->value map in \p
 * valueMapPtr. If the original net had several values with the same name, the
 * map will contain the value for the last definition. valueMapPtr is optional.
 * \p Prefix can be used for appending some string to every operator name (e.g.
 * we can add "caffe2::").
 */
TORCH_API void convertNetDefToIR(
    const caffe2::NetDef& net,
    Graph* graph,
    std::unordered_map<std::string, Value*>* valueMapPtr = nullptr,
    const std::string& prefix = "");

/** \brief Convert PyTorch IR \p graph to Caffe2 NetDef \p net.
 *
 * Note: for constant nodes (prim::Const) we generate a separate op in the net,
 * which might or might not be what we want. The idea here is that eventually
 * both formats will converge to PyTorch IR, so for now we try to keep as close
 * to it as possible. For short-term applications we might add a separate pass
 * that would fold such const-nodes into their users.
 * \p If Prefix is specified, the prefix will be removed from operator name when
 * converting from IR to NetDef.
 *
 * TODO: We might need to do a better job at preserving names of the variables,
 * especially external_inputs/external_outputs.
 */
TORCH_API void convertIRToNetDef(
    caffe2::NetDef* net,
    const Graph& graph,
    const std::string& prefix = "");

} // namespace jit
} // namespace torch
