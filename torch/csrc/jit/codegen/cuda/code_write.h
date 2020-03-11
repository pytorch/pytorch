#pragma once

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/iriostream.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <map>
#include <ostream>
#include <stack>

namespace torch {
namespace jit {
namespace fuser {

/*
std::ostream& operator<<(std::ostream& os, std::vector<Int*> vec) {
  os << "<";
  for (int i = 0; i < vec.size(); i++) {
    IRPrinter(os).print_inline(vec[i]);
    if (i == vec.size() - 1)
      os << ">";
    else
      os << ",";
  }
  return os;
}
*/

// Run through and grab all values that are used in this fusion based on
// the registered outputs.
struct FindUsedVals : public IterVisitor {
  std::vector<Val*> used_vals;

  void handle(Val* v) {
    used_vals.emplace_back(v);
  }

 public:
  static std::vector<Val*> find() {
    FindUsedVals finder;
    finder.traverse(FusionGuard::getCurFusion(), true);
    return finder.used_vals;
  }
};

struct TORCH_API CodeWrite : public IRPrinter {
 private:
  // Check if expr has a TensorView as an output
  bool isTVOp(const Expr* expr);

  /*****CODE PRINTING FUNCTIONS****/
  // Print the indexing into a TensorView
  void printIndexInto(std::vector<Int*> indices, const TensorView* const);
  // Compute and print the predicate based on accessing a specific TensorView
  bool print_predicate(const TensorView* const);

  // Print the allocation of a register space
  void printAlloc(TensorView*);
  // Print lhs of uop/bop, returns if predicate was needed
  bool printConsumer(TensorView*);
  // Printing functions for TensorView ops
  void handle(const TensorView* const);
  // Check overrides before printing a value
  void handle(const Val* const);
  void handle(const UnaryOp* const);
  void handle(const BinaryOp* const);

  /****END CODE PRINTING FUNCTIONS****/

  // Ignore split/merge/reorder operations,
  // we don't want to print them.
  void handle(const Split* const) {}
  void handle(const Merge* const) {}
  void handle(const Reorder* const) {}

  // Indent the generated code
  void indent();

  // Update the for loop structure based on provided TensorView output
  void updateView(TensorView*);

  // Grab all the indices used in the current for loop structure
  std::vector<Int*> getLoopIndices();
  // Open a new inner most for loop
  void openFor(IterDomain*);
  // Close the inner most for loop
  void closeFor();
  // Close all for loops
  void resetFors();
  // Clear out the last recorded computeAtView
  void clearActiveView();

  // Mark if the TensorView I'm printing is a producer
  bool producer = false;
  // Track the TensorView that is consuming the current producer
  TensorView* consumer = nullptr;

  // Track the for loops
  std::vector<const ForLoop*> fors;
  // Track the indentation size for pretty printing
  int indent_size = 0;

  // Track the last computeAt TensorView and axis
  const TensorView* active_view = nullptr;
  int active_view_axis = 0;

  // Mark if I want to reset all fors next time I call updateView
  bool reset_fors = false;

  // Track all bound iter domains
  std::set<IterDomain*> bound_iters;

  // Print std::string instead of Val
  std::map<const Val* const, std::string> overrides;

  // Set override for thread/block axes
  void bind(IterDomain* id, Val* iterator);

  // Grab all values that are used. Look for TensorViews setting the overrides
  // maps for Int* -> Tensor->size(i) Grab all IterDoms that are used. Look for
  // any mappings to threadIdx.x / blockIdx.x add them to bound_iters
  void setupOverrides();

  // wrapper for overrides.find on non-const vals
  std::map<const Val* const, std::string>::iterator overrides_find(Val* val) {
    return overrides.find(const_cast<const Val* const>(val));
  }
  // wrapper for override.emplace on non-const vals
  void overrides_emplace(Val* val, std::string str) {
    overrides[const_cast<const Val* const>(val)] = str;
  }

  // Print the header of the kernel
  void header();

 public:
  // Init printer on ostream
  CodeWrite(std::ostream& _os, std::string kernel_name = "kernel")
      : IRPrinter(_os), kernel_name_(kernel_name) {}

  // print generated code to ostream
  void traverse(Fusion* fusion);
  std::string kernel_name_;
};

} // namespace fuser
} // namespace jit
} // namespace torch
