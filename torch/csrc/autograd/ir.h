#pragma once

// TODO: Remove Python dependency with layer of indirection

#include <Python.h>
#include <memory>
#include <vector>
#include <cassert>

#include "torch/csrc/utils/object_ptr.h"

namespace torch { namespace autograd {

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

// This IR is based on administrative normal form.

class Arg;
using arg_list = std::vector<std::shared_ptr<Arg>>;

class Expr;

using Location = std::string;

// --------------------------------------------------------------------
// Arguments, which can be passed to functions.  NON-tupled.

class Arg {
public:
  enum class Id {
    Local, // also known as Variable, but not called that to avoid confusion
    PyConst
  };

  Id _id;
  Arg(Id id) : _id(id) {}

  Arg(const Arg& other) = delete;
  Arg(Arg&& other) = delete;
};

struct Local : public Arg {
  // This is not used at the moment (except in the invocation of
  // the Arg constructor, where it could be trivially inlined),
  // but it can be used to implement some useful, type-dispatched
  // functions.  See is(), dynCast() and cast() in
  // https://github.com/WebAssembly/binaryen/blob/master/src/wasm.h
  const static Id SelfId = Id::Local;
  int unique;
  Local(int unique)
    : Arg(SelfId)
    , unique(unique)
    {}
};
using local_list = std::vector<std::shared_ptr<Local>>;

struct PyConst : public Arg {
  const static Id SelfId = Id::PyConst;
  THPObjectPtr pyobj;
  PyConst(THPObjectPtr&& pyobj)
    : Arg(SelfId)
    , pyobj(std::move(pyobj))
    {}
};

// SubType is instance of CRTP; allows us to avoid virtual dispatch
// ParameterType is used in lieu of the member variable trick, because
// it allows you to do visitor tail calls.
template<typename SubType, typename ReturnType = void>
struct ArgVisitor {
  /*
  // Purposely undefined, to avoid C++ thinking that we are eventually
  // going to define these (we will not.)
  ReturnType visitLocal(std::shared_ptr<Local>, T...);
  ReturnType visitPyConst(std::shared_ptr<PyConst>, T...);
  */
  template <typename... T>
  ReturnType visitArg(std::shared_ptr<Arg> arg, T&&... args) {
    switch (arg->_id) {
      case Arg::Id::Local:
        return static_cast<SubType*>(this)->visitLocal(std::static_pointer_cast<Local>(arg), args...);
      case Arg::Id::PyConst:
        return static_cast<SubType*>(this)->visitPyConst(std::static_pointer_cast<PyConst>(arg), args...);
    }
    __builtin_unreachable();
  }
};

// --------------------------------------------------------------------
// Bindings

struct Bind {
  local_list lvals;
  std::shared_ptr<Expr> rval;
  Bind(local_list lvals, std::shared_ptr<Expr> rval)
    : lvals(lvals)
    , rval(rval)
    {}
};

// --------------------------------------------------------------------
// Expressions (returns tupled arguments)

struct Expr {
  enum class Id {
    PyApply,
    Let,
    Locals,
  };

  Id _id;
  Location _loc;
  Expr(Id id) : _id(id) {}
  Expr(Id id, Location loc) : _id(id), _loc(loc) {}

  Expr(const Expr& other) = delete;
  Expr(Expr&& other) = delete;
};

struct PyApply : public Expr {
  const static Id SelfId = Id::PyApply;
  THPObjectPtr pyobj;
  arg_list args;
  bool is_legacy;
  PyApply(THPObjectPtr&& pyobj, arg_list args, bool is_legacy, Location loc)
    : Expr(SelfId, loc)
    , pyobj(std::move(pyobj))
    , args(args)
    , is_legacy(is_legacy)
    {}
  PyApply(THPObjectPtr&& pyobj, arg_list args, bool is_legacy)
    : Expr(SelfId)
    , pyobj(std::move(pyobj))
    , args(args)
    , is_legacy(is_legacy)
    {}
};

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

struct Locals : public Expr {
  const static Id SelfId = Id::Locals;
  local_list locals;
  Locals(local_list locals, Location loc)
    : Expr(SelfId, loc)
    , locals(locals)
    {};
  Locals(local_list locals)
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
  ReturnType visitPyApply(std::shared_ptr<PyApply>, T...);
  ReturnType visitLet(std::shared_ptr<Let>, T...);
  */
  template <typename... T>
  ReturnType visitExpr(std::shared_ptr<Expr> e, T&&... args) {
    switch (e->_id) {
      case Expr::Id::PyApply:
        return static_cast<SubType*>(this)->visitPyApply(std::static_pointer_cast<PyApply>(e), args...);
      case Expr::Id::Let:
        return static_cast<SubType*>(this)->visitLet(std::static_pointer_cast<Let>(e), args...);
      case Expr::Id::Locals:
        return static_cast<SubType*>(this)->visitLocals(std::static_pointer_cast<Locals>(e), args...);
    }
    __builtin_unreachable();
  }
};

void printExpr(std::shared_ptr<Expr>);
void printExpr(std::shared_ptr<Expr>, std::ostream& s);

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
