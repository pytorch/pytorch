#pragma once

#include <ATen/core/ivalue.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <ir_base_nodes.h>
#include <ir_container.h>
#include <iter_visitor.h>

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
class KernelArgumentHolder;

//! Fusion Guard is our "context manager". It holds the actrive fusion and
//! allows it to be accessed anywhere through FusionGuard::getCurFusion()
class TORCH_CUDA_CU_API FusionGuard {
 public:
  Fusion* prev_fusion;

  //! Set the active fusion so it can be manipulated.
  explicit FusionGuard(Fusion* fusion);

  ~FusionGuard();

  static Fusion* getCurFusion();
  static void setCurFusion(Fusion* fusion);
};

//! Fusion is mutable but unique. Nodes cannot be copied in any way from one
//! Fusion to another. If anything like that is desired, it would require
//! duplicating all associated values and exprs. Fusion is considered to be SSA,
//! though this could also change in the future if there is a good reason to do
//! so.
//!
//! The Fusion owns the whole IR graph (Vals and Exprs)
//!
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API Fusion : public IrContainer {
  typedef std::unordered_map<int, std::vector<int64_t>> PermutationMap;

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
  void removeExpr(Expr* expr) override;

  //! Completely remove val from the fusion, break all dependencies associated
  //! with it
  void removeVal(Val* val) override;

  //! Register input as an input of the fusion
  void addInput(Val* input);

  //! Register output as an output of the fusion
  void addOutput(Val* output);

  //! Deregister input as an input of the fusion
  void removeInput(Val* input);

  //! Deregister output as an output of the fusion
  void removeOutput(Val* output);

  //! Replace output with another value
  void replaceOutput(Val* output, Val* replacement);

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
  void printKernel(DataType index_type = DataType::Int);

  //! Lower the fusion and evaluate bank conflict info
  std::unordered_map<std::string, std::pair<int, int>> bankConflictInfo(
      DataType index_type = DataType::Int);

  //! Return a list of topologically sorted expressions. This only includes
  //! exprs required to genereate registered outputs.
  std::vector<Expr*> exprs();

  //! Return a vector of fusion inputs that feed this Val
  std::vector<Val*> inputsOf(Val* val);

  //! Return all Vals in math expressions that cannot be eliminated.
  //!
  //! It is generally equivalent to vals that are used to generate
  //! outputs, however, when a multi-output expression exists, and only
  //! some of the outputs are used, the remaining unused outputs are
  //! also included as they must show up in the final code.
  std::vector<Val*> usedMathVals();

  //! Returns all vals that are produced by used math expressions and
  //!  also do not have further consumers.
  //!
  //! In the case of an active multi-output expressions, the returned vector
  //!  will include the expression outputs that did not lead to an fusion
  //!  output.
  std::vector<Val*> terminatingMathVals();

  //! Return all Exprs that use val
  std::unordered_set<Expr*> unordered_uses(const Val* val) const;

  //! Return the Expr that produces val
  Expr* definition(const Val* val) const;

  //! Indicate to kernel to set itself up to generate random numbers
  bool isStochastic();

  //! Run fusion segmentation algorithm to create a segmented fusion
  std::unique_ptr<SegmentedFusion> segment(const KernelArgumentHolder& args);

  const auto& inputs() const {
    return inputs_;
  }

  std::vector<Val*> inputsAndCreated();

  const auto& outputs() const {
    return outputs_;
  }

  std::vector<Val*> getTerminatingOutputs() const;

  // Aliasing output to input value, this is a WAR to allow inplace update on
  // input tensor.
  // Note: this is not always safe and should be used with extra caution.
  // Currently the only place it's used is in the running stats update for batch
  // normalization.
  // TODO: alias should be made aware to segmentation, so we'll always include
  // the input tensor to the section where output is produced.
  void aliasOutputToInput(Val* output, Val* input);
  Val* getOutputAlias(Val* output);
  std::unordered_set<int> getOutputAliasIndices() const;
  std::vector<std::pair<int, int>> getInputAliasIndices() const;

  // mark input at index to be permuted by permutation
  void setPermutationOnInput(int index, std::vector<int64_t> permutation) {
    permuted_input_map_.insert({index, permutation});
  }

  // mark output at index to be restored by permutation
  void setPermutationOnOutput(int index, std::vector<int64_t> permutation) {
    permuted_output_map_.insert({index, permutation});
  }

  // return a map of indices to permutation, which indicates all input tensors
  // that needs to be permuted
  const PermutationMap& getPermutationInputMap() const {
    return permuted_input_map_;
  }

  // return a map of indices to permutation, which indicates all output tensors
  // that needs to be permuted
  const PermutationMap& getPermutationOutputMap() const {
    return permuted_output_map_;
  }

  bool isTVUseInfoValid() {
    return all_tv_uses_valid_;
  }

  bool isUpdatingTVUseInfo() {
    return is_during_update_uses_;
  }

  const auto& ioAlias() const {
    return io_alias_;
  }

 protected:
  friend SegmentCandidateFinder;
  friend SegmentedFusion;
  friend class TranslateApplicableWelford;
  friend Val;

  static IrCloner copy(const Fusion* from, Fusion* to);

  //! Register the Val with this fusion
  virtual void registerVal(Val* val) override;

  //! Register expr with this fusion.
  //! When we register an expression, we want to update the dependency tracking
  //! of Vals. If this container is a not a Kernel, it will remove previous
  //! definitions of outputs and register this Expr as the definition. Otherwise
  //! will update definition if not previously set, but will not remove old
  //! definitions.
  virtual void registerExpr(Expr* expr) override;

  //! Clear Expr's from TV uses that are not required to produce outputs from
  //! inputs. Only other place this is used (other than Fusion) is in
  //! Val::uses()
  void resetTvUses();

 private:
  // Determine if the two values are compatible for aliasing
  // Same DataType, ValType, and number of dimensions
  bool isAliasCompatible(Val* left, Val* right);

 private:
  // Fusion inputs and outputs
  std::vector<Val*> inputs_;
  std::vector<Val*> outputs_;

  // io alias pointing from output to input
  std::unordered_map<Val*, Val*> io_alias_;

  // See Note [ Permutation support in nvfuser ]
  // map from indices of input tensor to permutation
  PermutationMap permuted_input_map_;
  // map from indices of output tensor to permutation
  PermutationMap permuted_output_map_;

  // Records if the current use data in the IR nodes are valid
  //  the states are either all valid or all invalid
  bool all_tv_uses_valid_ = false;
  bool is_during_update_uses_ = false;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
