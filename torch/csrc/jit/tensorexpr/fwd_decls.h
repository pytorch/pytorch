#pragma once
#include <c10/core/ScalarType.h>
#include <memory>

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename Node>
using NodePtr = std::shared_ptr<Node>;

template <typename To, typename From>
NodePtr<To> to(NodePtr<From> x) {
  return std::dynamic_pointer_cast<To>(x);
}

template <typename To, typename From>
NodePtr<To> static_to(NodePtr<From> x) {
  return std::static_pointer_cast<To>(x);
}

template <typename Node, typename... Args>
NodePtr<Node> alloc(Args&&... args) {
  return std::make_shared<Node>(std::forward<Args>(args)...);
}

class Buf;
class Expr;
class Stmt;
class Var;

using BufPtr = NodePtr<Buf>;
using ExprPtr = NodePtr<Expr>;
using StmtPtr = NodePtr<Stmt>;
using VarPtr = NodePtr<Var>;

class ExprHandle;
class VarHandle;
class BufHandle;

class Add;
class And;
class BitCast;
class Broadcast;
class Cast;
class CompareSelect;
class Div;
class IfThenElse;
class Intrinsics;
class Let;
class Load;
class Lshift;
class Max;
class MaxTerm;
class Min;
class MinTerm;
class Mod;
class Mul;
class Or;
class Polynomial;
class Ramp;
class ReduceOp;
class RoundOff;
class Rshift;
class Store;
class Sub;
class Term;
class Xor;
using AddPtr = NodePtr<Add>;
using AndPtr = NodePtr<And>;
using BitCastPtr = NodePtr<BitCast>;
using BroadcastPtr = NodePtr<Broadcast>;
using CastPtr = NodePtr<Cast>;
using CompareSelectPtr = NodePtr<CompareSelect>;
using DivPtr = NodePtr<Div>;
using IfThenElsePtr = NodePtr<IfThenElse>;
using IntrinsicsPtr = NodePtr<Intrinsics>;
using LetPtr = NodePtr<Let>;
using LoadPtr = NodePtr<Load>;
using LshiftPtr = NodePtr<Lshift>;
using MaxPtr = NodePtr<Max>;
using MaxTermPtr = NodePtr<MaxTerm>;
using MinPtr = NodePtr<Min>;
using MinTermPtr = NodePtr<MinTerm>;
using ModPtr = NodePtr<Mod>;
using MulPtr = NodePtr<Mul>;
using OrPtr = NodePtr<Or>;
using PolynomialPtr = NodePtr<Polynomial>;
using RampPtr = NodePtr<Ramp>;
using ReduceOpPtr = NodePtr<ReduceOp>;
using RoundOffPtr = NodePtr<RoundOff>;
using RshiftPtr = NodePtr<Rshift>;
using StorePtr = NodePtr<Store>;
using SubPtr = NodePtr<Sub>;
using TermPtr = NodePtr<Term>;
using XorPtr = NodePtr<Xor>;

class Allocate;
class AtomicAdd;
class Block;
class Cond;
class ExternalCall;
class For;
class Free;
class SyncThreads;
using AllocatePtr = NodePtr<Allocate>;
using AtomicAddPtr = NodePtr<AtomicAdd>;
using BlockPtr = NodePtr<Block>;
using CondPtr = NodePtr<Cond>;
using ExternalCallPtr = NodePtr<ExternalCall>;
using ForPtr = NodePtr<For>;
using FreePtr = NodePtr<Free>;
using SyncThreadsPtr = NodePtr<SyncThreads>;

#define IMM_DECLARE(Type, Name) \
  class Name##Imm;              \
  using Name##ImmPtr = NodePtr<Name##Imm>;
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_DECLARE);
#undef IMM_DECLARE

} // namespace tensorexpr
} // namespace jit
} // namespace torch
