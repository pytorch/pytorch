#include <Python.h>
#include "torch/csrc/jit/ir.h"
#include <toffee/toffee.pb.h>
#include <vector>

namespace torch { namespace autograd {

struct PrimSpecContext {
  // TODO: For now, this is hardcoded to write protobufs.  Possibly add
  // an abstraction barrier here...
  toffee::GraphProto* graph;
  // TODO: add fresh name supply

  // Get the Toffee name for a node in the IR
  // TODO: add name conversion mapping (currently hardcoded using uniques from
  // Node)
  std::string node(jit::Node *n) {
    return std::to_string(n->unique());
  }
};

struct primspec_unconvertible : public std::runtime_error {
  // la la la no constructor inheritance
  explicit primspec_unconvertible(const std::string& message) : std::runtime_error(message) {}
};

struct HasPrimSpec {
  // Add some nodes to the Toffee protobuf, under the assumption that this node
  // as a whole has the represented inputs and outputs.  Raises a
  // primspec_unconvertible exception if conversion is not supported.
  //
  // TODO: Passing in Node* directly here feels like too tight coupling with the
  // IR.  Changing this to variable_list will reduce coupling (but it means we
  // can only run these AS we are executing.)
  virtual void primspec(PrimSpecContext* ctx, jit::node_list inputs, jit::node_list outputs) = 0;
};

}} // namespace torch::autograd
