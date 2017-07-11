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
// operator, and you don't have to recompile PyTorch.  So... Zach is
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
struct Local;

// Something that takes some number of tensor inputs, and produces some
// number of tensor outputs.  The "prim-ops", so to speak.
struct Operator;

// Something that produces some number of tensor outputs.  In ANF
// formalism, this is often called a CExpr.
struct Instruction;

// The top-level expression that handles bindings.  This is basically
// a glorified singly linked list with a return statement at the end.
// But in the functional interpretation, its a series of let-bindings
// until you get to the end.  It's not very efficient.
struct Expr;

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

using local_list = std::vector<std::shared_ptr<Local>>;
using pyobj_list = std::vector<THPObjectPtr>;
using Location = std::string;

// --------------------------------------------------------------------
// Variables, which refer to tensors (NEVER tuples of tensors)

struct Local {
  int unique;
  Local(int unique)
    : unique(unique)
    {}
};

// --------------------------------------------------------------------
// Operators, which map tensors to tensors

// Although there is only one choice at the moment, this is the primary
// candidate for adding extra options.  At the moment this AST is designed
// CLOSED but long term we need a solution for operator extension.  That
// requires careful design, because whatever it is we request from users
// when they implement operator, is what we get FOREVER.

struct Operator {
  enum class Id {
    PythonOp,
  };
  Id _id;
  Operator(Id id) : _id(id) {}

  Operator(const Operator& other) = delete;
  Operator(Operator&& other) = delete;
};

// The calling convention for a Python function specifies how we
// need to intersperse tensor and Python arguments.
struct PyFunctionCConv {
  enum class ArgType { Tensor, Scalar };
  std::vector<ArgType> arg_types;
  PyFunctionCConv(std::vector<ArgType> arg_types) : arg_types(arg_types) {}
};

struct PythonOp : public Operator {
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
  PyFunctionCConv cconv;
  // Whether or not this is a legacy function implementation or not.
  // We don't support executing traces with legacy functions at the moment.
  bool is_legacy;
  // Scalar arguments to the Python function.  Not necessarily passed to
  // the function in this order; see cconv for the correct order.
  std::vector<THPObjectPtr> scalar_args;
  // NB: tensor arguments are in Apply
  PythonOp(THPObjectPtr&& pyobj, PyFunctionCConv cconv, bool is_legacy, pyobj_list&& scalar_args)
    : Operator(SelfId)
    , pyobj(std::move(pyobj))
    , cconv(cconv)
    , is_legacy(is_legacy)
    , scalar_args(std::move(scalar_args))
    {}
};

// SubType is instance of CRTP; allows us to avoid virtual dispatch
template<typename SubType, typename ReturnType = void>
struct OperatorVisitor {
  /*
  // Purposely undefined, to avoid C++ thinking that we are eventually
  // going to define these (we will not.)
  ReturnType visitPythonOp(std::shared_ptr<PythonOp>, T...);
  */
  template <typename... T>
  ReturnType visitOperator(std::shared_ptr<Operator> o, T&&... args) {
    switch (o->_id) {
      case Operator::Id::PythonOp:
        return static_cast<SubType*>(this)->visitPythonOp(std::static_pointer_cast<PythonOp>(o), args...);
    }
    __builtin_unreachable();
  }
};


// --------------------------------------------------------------------
// Instruction

// It is possible that not making this a variant is a mistake.  I am
// not sure.  The benefit of making this a variant is that primitive
// operators can be given instructions where the number of locals is
// statically known.

struct Instruction {
  std::shared_ptr<Operator> op;
  local_list args;
  Location loc;

  Instruction(std::shared_ptr<Operator> op, local_list args, Location loc)
    : op(op)
    , args(args)
    , loc(loc)
    {}
  Instruction(std::shared_ptr<Operator> op, local_list args)
    : op(op)
    , args(args)
    , loc("")
    {}
};

// --------------------------------------------------------------------
// Expressions (returns tupled arguments)

struct Expr {
  enum class Id {
    Let,
    Tuple,
  };

  Id _id;
  Location _loc;
  Expr(Id id) : _id(id) {}
  Expr(Id id, Location loc) : _id(id), _loc(loc) {}

  Expr(const Expr& other) = delete;
  Expr(Expr&& other) = delete;
};

// Helper structure for bindings.  Having a separate struct could come in handy
// if we ever support recursive bindings (probably not), or we just want to put
// all the bindings in a giant vector.
struct Bind {
  local_list lvals;
  std::shared_ptr<Instruction> rval;
  Bind(local_list lvals, std::shared_ptr<Instruction> rval)
    : lvals(lvals)
    , rval(rval)
    {}
};

// AKA singly linked list
struct Let : public Expr {
  const static Id SelfId = Id::Let;
  Bind bind;
  std::shared_ptr<Expr> expr;
  Let(Bind bind, std::shared_ptr<Expr> expr)
    : Expr(SelfId)
    , bind(bind)
    , expr(expr)
    {}
};

// AKA return
struct Tuple : public Expr {
  const static Id SelfId = Id::Tuple;
  local_list locals;
  Tuple(local_list locals, Location loc)
    : Expr(SelfId, loc)
    , locals(locals)
    {};
  Tuple(local_list locals)
    : Expr(SelfId)
    , locals(locals)
    {};
};

// SubType is instance of CRTP; allows us to avoid virtual dispatch
template<typename SubType, typename ReturnType = void>
struct ExprVisitor {
  /*
  // Purposely undefined, to avoid C++ thinking that we are eventually
  // going to define these (we will not.)
  ReturnType visitLet(std::shared_ptr<Let>, T...);
  ReturnType visitTuple(std::shared_ptr<Tuple>, T...);
  */
  template <typename... T>
  ReturnType visitExpr(std::shared_ptr<Expr> e, T&&... args) {
    switch (e->_id) {
      case Expr::Id::Let:
        return static_cast<SubType*>(this)->visitLet(std::static_pointer_cast<Let>(e), args...);
      case Expr::Id::Tuple:
        return static_cast<SubType*>(this)->visitTuple(std::static_pointer_cast<Tuple>(e), args...);
    }
    __builtin_unreachable();
  }
};

void printExpr(std::shared_ptr<Expr>);
void printExpr(std::shared_ptr<Expr>, std::ostream& s);

// --------------------------------------------------------------------
// Graphs (i.e., functions, i.e., lambda expressions)

struct Graph {
  local_list params;
  std::shared_ptr<Expr> body;
  Graph(local_list params, std::shared_ptr<Expr> body)
    : params(params)
    , body(body)
    {};
};

void printGraph(std::shared_ptr<Graph>, std::ostream& s);

// --------------------------------------------------------------------
// IR builder

// This builder allows you to build a sequence of successively nested
// let statements
//
// TODO: Consider doing liveness analysis at this point

class LetBuilder {
  std::vector<Bind> binds;
public:
  LetBuilder() {};
  void add(Bind bind) { binds.emplace_back(bind); }
  std::shared_ptr<Expr> expr(std::shared_ptr<Expr> e) {
    for (auto it = binds.rbegin(); it != binds.rend(); ++it) {
      e = std::make_shared<Let>(*it, e);
    }
    return e;
  }
};

}}
