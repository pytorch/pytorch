#ifndef NOM_REPRESENTATIONS_COMPILER_H
#define NOM_REPRESENTATIONS_COMPILER_H

#include "caffe2/core/common.h"
#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Support/Casting.h"

namespace nom {
namespace repr {

class TORCH_API Value {
 public:
  enum class ValueKind { Value, Instruction, Data };
  Value(ValueKind K) : kind_(K) {}
  Value() : kind_(ValueKind::Value) {}
  ValueKind getKind() const {
    return kind_;
  }
  virtual ~Value() = default;

 private:
  const ValueKind kind_;
};

class TORCH_API Data : public Value {
 public:
  Data() : Value(ValueKind::Data) {}
  static bool classof(const Value* V) {
    return V->getKind() == ValueKind::Data;
  }
  virtual ~Data() = default;
  size_t getVersion() const {
    return version_;
  }

  void setVersion(size_t version) {
    version_ = version;
  }

 private:
  size_t version_ = 0;
};

class TORCH_API Instruction : public Value {
 public:
  /// \brief All the different types of execution.
  enum class Opcode {
    Generic, // Handles basic instructions.
    TerminatorStart, // LLVM style range of operations.
    Branch,
    Return,
    TerminatorEnd,
    Phi
  };
  Instruction() : Value(ValueKind::Instruction), op_(Opcode::Generic) {}
  Instruction(Opcode op) : Value(ValueKind::Instruction), op_(op) {}
  static bool classof(const Value* V) {
    return V->getKind() == ValueKind::Instruction;
  }
  virtual ~Instruction() = default;
  Opcode getOpcode() const {
    return op_;
  }

 private:
  Opcode op_;
};

class TORCH_API Terminator : public Instruction {
 public:
  Terminator(Instruction::Opcode op) : Instruction(op) {}

 private:
  static bool classof(const Value* V) {
    return isa<Instruction>(V) &&
        isTerminator(cast<Instruction>(V)->getOpcode());
  }
  static bool isTerminator(const Opcode& op) {
    return op >= Opcode::TerminatorStart && op <= Opcode::TerminatorEnd;
  }
};

class TORCH_API Branch : public Terminator {
 public:
  Branch() : Terminator(Instruction::Opcode::Branch) {}
};

class TORCH_API Return : public Terminator {
 public:
  Return() : Terminator(Instruction::Opcode::Return) {}
};

class TORCH_API Phi : public Instruction {
 public:
  Phi() : Instruction(Instruction::Opcode::Phi) {}
};

} // namespace repr
} // namespace nom

#endif // NOM_REPRESENTATIONS_COMPILER_H
