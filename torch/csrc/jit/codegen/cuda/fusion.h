#pragma once

#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Usage: FusionGuard and Fusion are required user interfaces for any operation
//! underlying the code generator. In order to create values, expressions, and
//! generate code a Fusion instance must be active. It is the responsibility of
//! the user to create a Fusion instance and register it with the fusion guard.
//! The simplest example of this is:
//!
//!     Fusion fusion;
//!     FusionGuard fg(&fusion);
//!
//! Once a fusion is active all values and operations will be registered with
//! it.
//!
//! FusionGuard and Fusion are critical to the lifetime model of the IR system.
//! FusionGuard is a convenient way to set what base container instance holds
//! the defined IR. Statements that are defined are registered through the
//! FusionGuard with a particular Fusion. FusionGuard provides convenient
//! methods to access the active fusion so it doesn't need to be passed around
//! constantly. Any IR node derived classes from Statement must register with
//! Fusion to avoid memory leaks.
//!
//! Fusion is generally thought of as a translated fusion group from the JIT. It
//! is likely a single kernel, although, we don't have to stick to this in the
//! future and could in theory generate multiple kernels with an executor to run
//! them.
//!
//! Fusion also allows users to set input/output values that will allow us to
//! figure out how to hook up runtime data to and from the JIT as well as
//! provide us mechanisms for dependency analysis and DCE including safety
//! checks.

class Fusion;
class TensorView;
class WelfordResult;

class SegmentCandidateFinder;
class SegmentedFusion;

//! Fusion Guard is our "context manager". It holds the actrive fusion and
//! allows it to be accessed anywhere through FusionGuard::getCurFusion()
class TORCH_CUDA_CU_API FusionGuard {
 public:
  Fusion* prev_fusion;

  //! Set the active fusion so it can be manipulated.
  explicit FusionGuard(Fusion* fusion);

  ~FusionGuard();

  static Fusion* getCurFusion();
};

//! Fusion is mutable but unique. Nodes cannot be copied in any way from one
//! Fusion to another. If anything like that is desired, it would require
//! duplicating all associated values and exprs. Fusion is considered to SSA,
//! though this could also change in the future if there is a good reason to do
//! so.
//!
//! The Fusion owns the whole IR graph (Vals and Exprs)
//!
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API Fusion final {
 public:
  Fusion() = default;

  Fusion(const Fusion& other);
  Fusion(Fusion&& other) noexcept;

  Fusion& operator=(const Fusion& other);
  Fusion& operator=(Fusion&& other) noexcept;

  ~Fusion();

  friend void swap(Fusion& a, Fusion& b) noexcept;

  void clear() noexcept;

  //! Break dependency chains associated with Expr, remove references to expr
  //! delete expr
  void removeExpr(Expr* expr);

  //! Completely remove val from the fusion, break all dependencies associated
  //! with it
  void removeVal(Val* val);

  //! Register input as an input of the fusion
  // TODO: Rename to register
  void addInput(Val* input);

  //! Register output as an output of the fusion
  // TODO: Rename to register
  void addOutput(Val* output);

  //! Register output as an output of the fusion
  // TODO: Rename to register
  void addOutput(WelfordResult& output);

  //! Deregister input as an input of the fusion
  // TODO: Rename to register
  void removeInput(Val* input);

  //! Deregister output as an output of the fusion
  // TODO: Rename to register
  void removeOutput(Val* output);

  //! Replace output with another value
  void replaceOutput(Val* output, Val* replacement);

  //! Clear Expr's from TV uses that are not required to produce outputs from
  //! inputs
  void resetTvUses();

  //! Check if stmt is properly registered with this fusion
  bool inFusion(const Statement* stmt) const;

  //! Throw an error if stmt is not in this fusion
  void assertInFusion(const Statement* stmt, const std::string& msg = "") const;

  //! Assert that all leaves found from outputs are registered as an input
  void validateInputs();

  //! Print this fusion to the console
  void print();

  //! Print Arith exprs
  //! \param from_outputs_only Only print exprs reachable from outputs
  void printMath(bool from_outputs_only = true);

  //! Print transformations used in fusion (can be very verbose)
  void printTransforms();

  //! Lower the fusion and print a kernel
  void printKernel();

  //! Register the Val with this fusion
  StmtNameType registerVal(Val* val);

  //! Register expr with this fusion.
  //! When we register an expression, we want to update the dependency tracking
  //! of Vals. We add expr to our general expr_set_,
  StmtNameType registerExpr(Expr* expr);

  //! Register stmt with this fusion
  StmtNameType registerStatement(Statement* stmt);

  //! Return a list of topologically sorted expressions. This only includes
  //! exprs required to genereate registered outputs.
  std::vector<Expr*> exprs();

  //! Return a vector of fusion inputs that feed this Val
  std::vector<Val*> inputsOf(Val* val);

  //! Return the set of Vals registered with this fusion
  const std::unordered_set<Val*>& vals() const noexcept;

  //! Return in insertion order
  const std::deque<Val*>& deterministic_vals() const noexcept;

  //! Return all Vals in math expressions that cannot be eliminated.
  //!
  //! It is generally equivalent to vals that are used to generate
  //! outputs, however, when a multi-output expression exists, and only
  //! some of the outputs are used, the remaining unused outputs are
  //! also included as they must show up in the final code.
  std::vector<Val*> usedMathVals();

  //! Return the set of Exprs registered with this fusion. Warning: This will
  //! return exprs outside inputs/outputs, so can be unsafe for use with
  //! segmented fusions.
  const std::unordered_set<Expr*>& unordered_exprs() const noexcept;

  //! Return all Exprs that use val
  std::unordered_set<Expr*> unordered_uses(Val* val) const;

  //! Return the Expr that produces val
  Expr* definition(const Val* val) const;

  //! Indicate to kernel to set itself up to generate random numbers
  bool isStochastic();

  //! Indicate that the fusion contains reduction operations
  bool hasReduction();

  //! Indicate that the fusion contains welford operations
  bool hasWelford();

  //! Run fusion segmentation algorithm to create a segmented fusion
  std::unique_ptr<SegmentedFusion> segment(
      const at::ArrayRef<at::IValue>& inputs);

  const auto& inputs() const {
    return inputs_;
  }

  const auto& outputs() const {
    return outputs_;
  }

  std::vector<Val*> getTerminatingOutputs();

  bool hasInput(const Val* val) const;
  bool hasOutput(const Val* val) const;

  // Aliasing output to input value, this is a WAR to allow inplace update on
  // input tensor.
  // Note: this is not always safe and should be used with extra caution.
  // Currently the only place it's used is in the running stats update for batch
  // normalization.
  // TODO: alias should be made aware to segmentation, so we'll always include
  // the input tensor to the section where output is produced.
  void aliasOutputToInput(Val* output, Val* input);
  std::unordered_set<int> getOutputAliasIndices() const;
  std::vector<std::pair<int, int>> getInputAliasIndices() const;

  bool isTVUseInfoValid() {
    return all_tv_uses_valid_;
  }

  bool isUpdatingTVUseInfo() {
    return is_during_update_uses_;
  }

 protected:
  friend SegmentCandidateFinder;
  friend SegmentedFusion;
  friend class TranslateApplicableWelford;

  static IrCloner copy(const Fusion* from, Fusion* to);

 private:
  // Return an int that monotonically increases for each val/expr, some are
  // explicitly incremented by type.
  StmtNameType getValName(ValType vtype);
  StmtNameType getExprName();

  // Determine if the two values are compatible for aliasing
  // Same DataType, ValType, and number of dimensions
  bool isAliasCompatible(Val* left, Val* right);

 private:
  // Sets of all Vals/Exprs registered with this fusion
  // (val_deque_ is not owning the objects)
  std::unordered_set<Val*> val_set_;
  std::deque<Val*> val_deque_;
  std::unordered_set<Expr*> expr_set_;

  // Values names counters
  std::unordered_map<ValType, StmtNameType, TypeHash> val_type_name_map_;

  // Expression names counter
  StmtNameType expr_name_counter_ = 0;

  // Fusion inputs and outputs
  std::vector<Val*> inputs_;
  std::vector<Val*> outputs_;

  // io alias pointing from output to input
  std::unordered_map<Val*, Val*> io_alias_;

  // Records if the current use data in the IR nodes are valid
  //  the states are either all valid or all invalid
  bool all_tv_uses_valid_ = false;
  bool is_during_update_uses_ = false;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
