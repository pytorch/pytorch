#ifndef NOM_REPRESENTATIONS_COMPILER_H
#define NOM_REPRESENTATIONS_COMPILER_H

#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Support/Casting.h"

namespace nom {
namespace repr {

class Value {
public:
  enum class ValueKind { Value, Instruction, Data };
  Value(ValueKind K) : Kind(K) {}
  Value() : Kind(ValueKind::Value) {}
  ValueKind getKind() const { return Kind; }
  virtual ~Value() = default;

private:
  const ValueKind Kind;
};

class Data : public Value {
public:
  Data() : Value(ValueKind::Data) {}
  static bool classof(const Value *V) {
    return V->getKind() == ValueKind::Data;
  }
  virtual ~Data() = default;
  size_t getVersion() const { return Version; }

  void setVersion(size_t version) { Version = version; }

private:
  size_t Version = 0;
};

class Instruction : public Value {
public:
  /// \brief All the different types of execution.
  enum class Opcode {
    Generic,         // Handles basic instructions.
    TerminatorStart, // LLVM style range of operations.
    Branch,
    Return,
    TerminatorEnd,
    Phi
  };
  Instruction() : Value(ValueKind::Instruction), Op(Opcode::Generic) {}
  Instruction(Opcode op) : Value(ValueKind::Instruction), Op(op) {}
  static bool classof(const Value *V) {
    return V->getKind() == ValueKind::Instruction;
  }
  virtual ~Instruction() = default;
  Opcode getOpcode() const { return Op; }

private:
  Opcode Op;
};

class Terminator : public Instruction {
public:
  Terminator(Instruction::Opcode op) : Instruction(op) {}

private:
  static bool classof(const Value *V) {
    return isa<Instruction>(V) &&
           isTerminator(cast<Instruction>(V)->getOpcode());
  }
  static bool isTerminator(const Opcode &op) {
    return op >= Opcode::TerminatorStart && op <= Opcode::TerminatorEnd;
  }
};

class Branch : public Terminator {
public:
  Branch() : Terminator(Instruction::Opcode::Branch) {}
};

class Return : public Terminator {
public:
  Return() : Terminator(Instruction::Opcode::Return) {}
};

class Phi : public Instruction {
public:
  Phi() : Instruction(Instruction::Opcode::Phi) {}
};

} // namespace repr
} // namespace nom

#endif // NOM_REPRESENTATIONS_COMPILER_H
