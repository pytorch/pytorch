#pragma once

#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

/*
 * Usage: FusionGuard and Fusion are required user interfaces for any operation
 * underlying the code generator. In order to create values, expressions, and
 * generate code a Fusion instance must be active. It is the responsibility of
 * the user to create a Fusion instance and register it with the fusion guard.
 * The simplest example of this is: Fusion fusion; FusionGuard fg(&fusion); Once
 * a fusion is active all values and operations will be registered with it.
 *
 * FusionGuard and Fusion are critical to the lifetime model of the IR system.
 * FusionGuard is a convenient way to set what base container instance holds the
 * defined IR. Statements that are defined are registered through the
 * FusionGuard with a particular Fusion. FusionGuard provides convenient methods
 * to access the active fusion so it doesn't need to be passed around
 * constantly. Any IR node derived classes from Statement must register with
 * Fusion to avoid memory leaks.
 *
 * Fusion is generally thought of as a translated fusion group from the JIT. It
 * is likely a single kernel, although, we don't have to stick to this in the
 * future and could in theory generate multiple kernels with an executor to run
 * them.
 *
 * Fusion also allows users to set input/output values that will allow us to
 * figure out how to hook up runtime data to and from the JIT as well as provide
 * us mechanisms for dependency analysis and DCE including safety checks.
 */

class Fusion;
class TensorView;

// Fusion Guard is our "context manager". It holds the actrive fusion and allows
// it to be accessed anywhere through FusionGuard::getCurFusion().
class TORCH_CUDA_CU_API FusionGuard {
 public:
  Fusion* prev_fusion;

  // Set the active fusion so it can be manipulated.
  explicit FusionGuard(Fusion* fusion);

  ~FusionGuard();

  static Fusion* getCurFusion();
};

/*
 * Fusion is mutable but unique. Nodes cannot be copied in any way from one
 * Fusion to another. If anything like that is desired, it would require
 * duplicating all associated values and exprs. Fusion is considered to SSA,
 * though this could also change in the future if there is a good reason to do
 * so.
 *
 * The Fusion owns the whole IR graph (Vals and Exprs)
 */
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

  // Break dependency chains associated with Expr, remove references to expr
  // delete expr.
  void removeExpr(Expr* expr);

  // Completely remove val from the fusion, break all dependencies associated
  // with it.
  void removeVal(Val* val);

  // Register input as an input of the fusion
  void addInput(Val* input);

  // Register output as an output of the fusion
  void addOutput(Val* output);

  // Check if stmt is properly registered with this fusion
  bool inFusion(const Statement* stmt) const;

  // Throw an error if stmt is not in this fusion. Message will be:
  // msg + " it was not found in the active fusion."
  void assertInFusion(const Statement* stmt, const std::string& msg = "") const;

  /*
   * Return a list of topologically sorted expressions. We can start
   * by only traversing back from registered outputs, or from all terminating
   * Vals.
   *
   * from_outputs_only:
   *   True - Sort from DAG associated with registered outputs
   *   False - Sort from all terminating Vals.
   */
  std::vector<Expr*> exprs(bool from_outputs_only = false);

  // Return a vector of fusion inputs that feed this Val
  std::unordered_set<Val*> inputsOf(Val* val);

  // Assert that all leaves found from outputs are registered as an input.
  void validateInputs();

  // Print this fusion to cout.
  void print();

  // Print Arith exprs used in outputs
  void printMath();

  // Print transformations used in fusion (can be very verbose)
  void printTransforms();

  // Lower the fusion and print a kernel
  void printKernel();

  // Register the Val with this fusion
  StmtNameType registerVal(Val* val);

  // Register expr with this fusion.
  // When we register an expression, we want to update the dependency tracking
  // of Vals. We add expr to our general expr_set_, we add use tracking for
  // inputs and origin tracking for outputs.
  StmtNameType registerExpr(Expr* expr);

  // Register stmt with this fusion.
  StmtNameType registerStatement(Statement* stmt);

  // Lowered nodes
  // TODO(kir): to be removed
  StmtNameType registerLoweredVal(Val* val);
  StmtNameType registerLoweredExpr(Expr* expr);

  // Lowered counterpart to inFusion()
  // TODO(kir): to be removed
  bool inKernelIr(const Statement* stmt) const;

  // Check if val is used in this fusion. Not equivelent to DCE
  bool used(Val* val) const;

  // Return the set of Vals registered with this fusion
  const std::unordered_set<Val*>& vals() const noexcept;
  // Return in insertion order
  const std::deque<Val*>& deterministic_vals() const noexcept;

  // Return the set of Exprs registered with this fusion
  const std::unordered_set<Expr*>& unordered_exprs() const noexcept;

  // Return all Exprs that use val
  std::unordered_set<Expr*> unordered_uses(Val* val) const;

  // Return the Expr that produces val
  Expr* origin(const Val* val) const;

  // Indicate to kernel to set itself up to generate random numbers
  bool isStochastic();

  // TODO(kir): revisit to see how many of these are still needed
  bool hasReduction();
  bool hasBlockReduction();
  bool hasGridReduction();
  bool hasBlockBroadcast();
  bool hasBroadcast();
  size_t gridReductionTempBufferSize();

  const auto& inputs() const {
    return inputs_;
  }

  const auto& outputs() const {
    return outputs_;
  }

  std::vector<Val*> getTerminatingOutputs();

  bool hasInput(const Val* val) const;
  bool hasOutput(const Val* val) const;

  void replaceInput(Val* replace, Val* with);
  void replaceOutput(Val* replace, Val* with);

 private:
  // Return an int that monotonically increases for each val/expr, some are
  // explicitly incremented by type.
  StmtNameType getValName(ValType vtype);
  StmtNameType getExprName();

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

  // Dependency tracking for Vals. Where did it come from? Where is it used?
  std::unordered_map<const Val*, Expr*> origin_;
  std::unordered_map<Val*, std::unordered_set<Expr*>> uses_;

  // Fusion inputs and outputs
  std::vector<Val*> inputs_;
  std::vector<Val*> outputs_;

  // Lowered IR
  std::unordered_set<Val*> lowered_val_set_;
  std::unordered_set<Expr*> lowered_expr_set_;
  std::unordered_map<const Val*, Expr*> lowered_origin_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
