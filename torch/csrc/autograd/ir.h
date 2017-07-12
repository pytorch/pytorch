#pragma once

// TODO: Remove Python dependency with layer of indirection

#include <Python.h>
#include <memory>
#include <vector>
#include <cassert>

#include "torch/csrc/utils/object_ptr.h"

namespace torch { namespace autograd {

// ---------------------------- >8 -------------------------------------
// Some comments on the IR:
//
// Variable bindings -> numbers, where tensors are stored
// Perhaps optimize down the slot space. But then the variable
// bindings are just numbers.  And we can change the list type
// from standard vector to interned first element, or if it is
// too big then it spills into malloc.  If you return one/two
// things, it's just a fixed size instruction, otherwise you have
// an overflow for the rest of the return types.
//
// It's nice to have the IR look like something that is similar
// to other interpreters, so that it's more familiar.
//
// Single-use is important... but the less encoded it is in the
// data structure, the easier it is to change.  Because the
// instructions could carry information about, e.g., "the data
// frame", rather than putting it in the IR.
//
// If you choose to do a fusion: imagine a graph, draw a circle around the ops
// to fuse. Any outgoing edges are live, and need to be written out when
// you fuse, but you can still do it in one pass, even when something is
// not completely consumed.  It's not a prerequisite, it is just part
// of the cost function for what the benefit would be.
//
// Get shapes for things and store it in the IR.
//
// OK to just do greedy fusion. Here are some fusions, do some fusions with it.
// Is there a DP framework where we just get the right answer automatically?
// Models for cost of doing optimization?  It's more stable: even if you
// scramble up the answer order, the answer doesn't change.  "Oh I refactor,
// this is prettier" and now performance goes away: NOT GOOD.
//
// Just recording global contiguousness might be good enough.
//
// Put the constants out of line, rather than interleaving them with the
// tensor arguments
//
// Short term: demonstrate what the best possible speedup is. (Use ATen)
// Subtlety: if you copy-pasted code and edit it, you have a modified
// node, and you don't have to recompile PyTorch.  So... Zach is
// curious with Python: how does the set of tools know where all your
// compilers are. So, hypothetically, setuptools should be able to "JIT"
// C++.  Write a hilarious C++ function as a subclass of function,
// and forward and backward is snippet of C++ code which are JIT
// compiled: take string, create shared library, dlopen.  And then you
// have that property.  Advantage to this, it's a test of using ATen
// with PyTorch.  Give us an idea of what speed of this compute is
// Python overhead, versus overhead in interpreter, or is it really
// fusion that is going to give us the speedup.
//
// And it's only five ops. And the ops aren't hard to dispatch.
// ---------------------------- >8 -------------------------------------

// A variable name, but we don't call it variable because that conflicts
// with PyTorch's existing use of Variable for autograd tracing.
struct Value;

// Something that takes some number of tensor inputs, and produces some
// number of tensor outputs.  The "prim-ops", so to speak.
struct Node;


// There are a bunch of ways to think about this IR, depending on your
// background.
//
// IF YOU ARE A FUNCTIONAL PROGRAMMER (ANF school): This is ANF, but we've named
// non let-binding expressions in a funny way as "instructions".  Otherwise,
// everything is as you expect.
//
// IF YOU ARE AN IMPERATIVE PROGRAMMER (SSA school): This is SSA, but the
// list of instructions and then the final return are named in a funny way
// as "expressions".  It's just the usual singly-linked list.
//
// Note that the IR as formulated today has NO control flow.  If we add it, I
// would strongly recommend the functional style representation, rather than SSA
// phi nodes.

using value_list = std::vector<std::shared_ptr<Value>>;
using node_list = std::vector<std::shared_ptr<Node>>;
using pyobj_list = std::vector<THPObjectPtr>;
using Location = std::string;

// --------------------------------------------------------------------
// Variables, which refer to tensors (NEVER tuples of tensors)

struct Value {
  int unique;
  Value(int unique)
    : unique(unique)
    {}
};

// --------------------------------------------------------------------
// Nodes, which map tensors to tensors

// Although there is only one choice at the moment, this is the primary
// candidate for adding extra options.  At the moment this AST is designed
// CLOSED but long term we need a solution for node extension.  That
// requires careful design, because whatever it is we request from users
// when they implement node, is what we get FOREVER.

struct Node {
  enum class Id {
    PythonOp,
  };
  value_list inputs;
  value_list outputs;
  Location loc;
  Id _id;
  Node(Id id) : _id(id) {}
  Node(const Node& other) = delete;
  Node(Node&& other) = delete;
  Id kind() {
    return _id;
  }
  Node & withLocation(const Location & loc_) {
    loc = loc_;
    return *this;
  }
};

struct PythonOp : public Node {
  // This is not used at the moment (except in the invocation of
  // the Arg constructor, where it could be trivially inlined),
  // but it can be used to implement some useful, type-dispatched
  // functions.  See is(), dynCast() and cast() in
  // https://github.com/WebAssembly/binaryen/blob/master/src/wasm.h
  const static Id SelfId = Id::PythonOp;
  // The Python object which contains the implementation of this function.
  // This is either a class (non-legacy) or an object (legacy).  See
  // TraceInterpreter for execution semantics.
  THPObjectPtr pyobj;
  // The calling convention for the Python function.
  // 's' -- python scalar argument
  // 't' -- tensor argument
  std::string cconv;
  // Whether or not this is a legacy function implementation or not.
  // We don't support executing traces with legacy functions at the moment.
  bool is_legacy;
  // Scalar arguments to the Python function.  Not necessarily passed to
  // the function in this order; see cconv for the correct order.
  std::vector<THPObjectPtr> scalar_args;
  // NB: tensor arguments are in Apply
  PythonOp(THPObjectPtr&& pyobj, const std::string & cconv, bool is_legacy, pyobj_list&& scalar_args)
    : Node(SelfId)
    , pyobj(std::move(pyobj))
    , cconv(cconv)
    , is_legacy(is_legacy)
    , scalar_args(std::move(scalar_args))
    {}
};

struct Graph {
  node_list nodes;
  value_list inputs;
  value_list outputs;
};
std::ostream& operator<<(std::ostream &  out, const Graph & g);

}}
