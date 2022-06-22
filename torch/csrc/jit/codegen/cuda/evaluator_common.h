#pragma once
#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <c10/core/DeviceType.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! This is the common space for expression evaluators in
//!  fusion IR and kernel IR context. Much of the evaluator
//!  optimizations and runtimes could share the same code
//!  path and they could be collected here.

class ExpressionEvaluator;

namespace kir {

class ExpressionEvaluator;

} // namespace kir

//! IR Contexts to be passed to generic evaluator optimizations
//!   and runtimes. Defines the essential interface for the
//!   generic logic to get necessary type and function info
//!   from the IR nodes. Generic optimizations will assume
//!   the same list of static definitions are provided
//!   in each of the contexts, just FusionIR and KernelIR
//!   currently.

//! Context for using generic logic on FusionIR
class FusionIRContext {
 public:
  using TV_TYPE = TensorView;
  using EVALUATOR_TYPE = ExpressionEvaluator;

  static BinaryOpType getOpType(BinaryOp* bop) {
    return bop->getBinaryOpType();
  }

  static UnaryOpType getOpType(UnaryOp* uop) {
    return uop->getUnaryOpType();
  }
};

//! Context for using generic logic on KernelIR
class KernelIRContext {
 public:
  using EVALUATOR_TYPE = kir::ExpressionEvaluator;

  static BinaryOpType getOpType(BinaryOp* bop) {
    return bop->getBinaryOpType();
  }

  static UnaryOpType getOpType(UnaryOp* uop) {
    return uop->getUnaryOpType();
  }
};

template <typename IRContext>
class PrecomputedIntegersBase;

//! NaiveIntegerMachine:
//!  This is an un-optimized runtime for evaluating a
//!   set of integers in one run. The runtime contains
//!   a vector of instructions inferred from IR at compile-time
//!   and it currently must be associated with an instance of
//!   PrecomputedIntegersBase that will provide the workspace
//!   containing the concrete values for the integers.
template <typename IRContext>
class NaiveIntegerMachine {
  //! The generic types of instructions supported for this
  //!  machine, currently only binary and unary.
  enum class InstructionType { UNARY_OP, BINARY_OP };

 public:
  //! Constructor lowers all the expr IR nodes stored in precomputed_integer
  //!  and stores them in the private state.
  NaiveIntegerMachine(PrecomputedIntegersBase<IRContext>& precomputed_integers);

  //! Runs all the instructions and write results to the associated
  //!  precomputed_integers.
  void run();

 private:
  //! Convert an unary IR expr to an instruction
  void makeUnaryOp(UnaryOp* uop);

  //! Convert an binary IR expr to an instruction
  void makeBinaryOp(BinaryOp* bop);

  //! Create an empty instruction with all default values
  //!  and place it at the end of the instruction buffer.
  int makeInstructionEntry();

  //! Run a single instruction at the given index of
  //!  the instruction buffer. Decodes and dispatches
  //!  to the corresponding instruction handle functions.
  void runInstruction(int index);

  //! Runs a unary operation at given index of instruction buffer
  void runUnaryOp(int index);

  //! Runs a binary operation at given index of instruction buffer
  void runBinaryOp(int index);

 private:
  friend PrecomputedIntegersBase<IRContext>;

  //! Reference to the PrecomputedInteger workspace associated with
  //!   this runtime. All the instructions will read and write the
  //!   values in this workspace.
  PrecomputedIntegersBase<IRContext>& precomputed_integers_;

  //! Instruction buffer. All states are in separate vectors and
  //!  the entry of each vector at the same index correspond to
  //!  the same instruction.

  //! Total number of instructions
  int num_of_instructions_ = 0;

  //! Machine instruction type for each instruction i.e.
  //!  unary or binary
  std::vector<InstructionType> inst_type_;

  //! Unary operator type if applicable, contains a default
  //!  value at each index corresponding to a binary op.
  std::vector<UnaryOpType> uop_type_;

  //! Unary operator type if applicable, contains a default
  //!  value at each index corresponding to a unary op.
  std::vector<BinaryOpType> bop_type_;

  //! Indexes of operands and destination of each instruction.
  //!  The indexes corresponds to positions in the workspace
  //!  where concrete values are hosted.

  //! Operand 0 of each instruction.
  std::vector<int> src0_;

  //! Operand 1 of each instruction, a default value at
  //!  each index corresponding to a unary op.
  std::vector<int> src1_;

  //! Destination of each instruction.
  std::vector<int> dest_;
};

//! PrecomputedIntegersBase:
//!  A class to support optimized evaluation of integers
//!  at runtime.
//!    At compile time all necessary integers are collected
//!  from given IR nodes and a runtime and a workspace containing
//!  the concrete values is created and pre-allocated.
//!    At runtime the integer vm is used to evaluate all the
//!  integers and store them in the workspace ahead of time.
template <typename IRContext>
class PrecomputedIntegersBase {
  using INTEGER_MACHINE = NaiveIntegerMachine<IRContext>;

 public:
  explicit PrecomputedIntegersBase() = default;

  //! Returns if the workspace contains evaluated results.
  bool ready() {
    return has_valid_values_;
  }

  //! Runs the internal integer machine that will compute
  //!  the values allocated in the workspace.
  void evaluate();

  //! Returns value for the given IR node if it's stored
  //!  in the workspace and has been evaluated.
  c10::optional<int64_t> getMaybeValueFor(const Val* val);

  //! Debugging helper, prints all the currently known values
  void print() const;

 protected:
  //! Initialize the workspace before first use.
  //!  Assume the given value list IR nodes have
  //!  been topologically sorted.
  void initializeValueList(
      typename IRContext::EVALUATOR_TYPE& evaluator,
      const std::vector<Val*>& sorted_value_list);

  //! Bind concrete value to the given index
  //!  if the index is valid.
  void bindValue(int index, int64_t value) {
    if (index < 0 || is_constant_[index]) {
      return;
    }
    defined_[index] = true;
    values_[index] = value;
    binding_log_.emplace_back(index, value);
  }

  //! Invalidate all computed values in the workspace.
  void invalidate();

  //! Interface for subclasses to access symbols_
  void loadSymbols(std::vector<Val*> symbols) {
    symbols_ = std::move(symbols);
  }

  //! Interface for subclasses to access symbols_
  std::vector<Val*>& symbols() {
    return symbols_;
  }

  //! Initialize the integer runtime that will
  //!  infer instructions from the workspace.
  void initializeIntegerMachine() {
    integer_machine_ = std::make_unique<INTEGER_MACHINE>(*this);
  }

  bool hasValidValues() {
    return has_valid_values_;
  }

 private:
  //! Post evaluation check, throws if any computed value
  //!  is inconsistent with its bound value
  void validate();

  //! Returns true if workspace has a computed or constant
  //!  value for given index.
  bool hasValue(int index) {
    TORCH_INTERNAL_ASSERT(index > 0);
    return defined_[index] || is_constant_[index];
  }

 private:
  friend INTEGER_MACHINE;

  //! Marks if an evaluation has finished
  bool has_valid_values_ = false;

  //! The size of workspace
  int num_of_values_ = -1;

  //! Marks if a value has been bound or
  //!  computed at each index.
  std::vector<bool> defined_;

  //! Marks if a value is compile-time constant
  //!  at each index.
  std::vector<bool> is_constant_;

  //! Stores the concrete values at each index.
  std::vector<int64_t> values_;

  //! Stores the IR nodes corresponding to each index.
  std::vector<Val*> symbols_;

  //! An internal log to keep track of all the bindings
  //!  used in each evaluation cycle. To be used for
  //!  consistency check.
  std::vector<std::pair<int, int64_t>> binding_log_;

  //! Integer runtime for realizing the integer computations.
  std::unique_ptr<INTEGER_MACHINE> integer_machine_;
};

//! PreComputedInteger workspace in Fusion IR context,
//!  defines the set of integers to be collected in each
//!  fusion graph and the input value binding given each
//!  fusion runtime input.
class FusionPrecomputedIntegers
    : public PrecomputedIntegersBase<FusionIRContext> {
  using precomputedIntegersBaseType = PrecomputedIntegersBase<FusionIRContext>;

 public:
  FusionPrecomputedIntegers(Fusion* fusion);

  //! Bind concrete values from fusion runtime inputs
  void bindFusionInputs(const at::ArrayRef<IValue>& aten_inputs);

 private:
  void bindTensorMetaData(TensorView* tv, const at::Tensor& at_tensor);

 private:
  Fusion* fusion_ = nullptr;
};
//! PreComputedInteger workspace in Fusion IR context,
//!  defines the set of integers to be collected in each
//!  kernel IR sequence and the input value binding given each
//!  fusion runtime input and launch constraints.
class KernelPrecomputedIntegers
    : public PrecomputedIntegersBase<KernelIRContext> {
  using precomputedIntegersBaseType = PrecomputedIntegersBase<KernelIRContext>;

 public:
  using ParallelExtentMap =
      std::unordered_map<ParallelType, std::vector<const Val*>, TypeHash>;

  KernelPrecomputedIntegers(kir::Kernel* kernel);

  //! Bind concrete values from fusion runtime inputs
  void bindKernelInputs(
      kir::Kernel* kernel,
      const at::ArrayRef<IValue>& aten_inputs);

  //! Bind concrete values from launch constraints
  void bindParallelExtents(
      const ParallelExtentMap& parallel_extents,
      const LaunchParams& launch_constraint);

  //! Bind the NamedScalars corresponding to the
  //!  concrete parallel dimension sizes after the
  //!  actual value has been resolved.
  void bindConcreteParallelTypeValue(ParallelType pt, int64_t value);

 private:
  void bindTensorMetaData(TensorView* tv, const at::Tensor& at_tensor);

  //! Iterate through all the named scalars corresponding
  //!  to thread sizes and pre-group them by their parallel
  //!  types.
  void initializeNamedScalars();

 private:
  //! Contains all the named scalars correspond
  //!  to thread size of each parallel type.
  std::unordered_map<ParallelType, std::unique_ptr<std::vector<int>>, TypeHash>
      thread_dim_value_indices_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
