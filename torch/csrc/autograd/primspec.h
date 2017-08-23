#pragma once

#ifdef WITH_TOFFEE
#include <Python.h>
#include "torch/csrc/jit/ir.h"
#include <toffee/toffee.pb.h>
#include <vector>
#endif

namespace torch { namespace autograd {

#ifdef WITH_TOFFEE

struct PrimSpecContext {
  std::shared_ptr<jit::Graph> graph;
  int batch_norm_count = 0;
};

struct primspec_unconvertible : public std::runtime_error {
  // la la la no constructor inheritance
  explicit primspec_unconvertible(const std::string& message) : std::runtime_error(message) {}
};
#endif // WITH_TOFFEE


struct HasPrimSpec {
#ifdef WITH_TOFFEE
  // Add some nodes to the Toffee protobuf, under the assumption that this node
  // as a whole has the represented inputs and outputs.  Raises a
  // primspec_unconvertible exception if conversion is not supported.
  //
  // TODO: Passing in Node* directly here feels like too tight coupling with the
  // IR.  Changing this to variable_list will reduce coupling (but it means we
  // can only run these AS we are executing.)
  virtual jit::node_list primspec(PrimSpecContext* ctx, jit::node_list inputs) = 0;
#endif // WITH_TOFFEE

// The reason we have this macro is because the class definition in headers need
// to declare primspec if they are overriding them... but this needs to be
// macro-controlled (don't define it if WITH_TOFFEE is not defined).  To make
// this work we use a macro; as an added bonus it's less typing.
#ifdef WITH_TOFFEE
#define HAS_PRIMSPEC virtual jit::node_list primspec(PrimSpecContext* ctx, jit::node_list inputs) override
#else // WITH_TOFFEE
#define HAS_PRIMSPEC
#endif // WITH_TOFFEE

};

}} // namespace torch::autograd
