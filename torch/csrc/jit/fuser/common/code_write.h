#pragma once

#include <torch/csrc/jit/fuser/common/arith.h>
#include <torch/csrc/jit/fuser/common/index_compute.h>
#include <torch/csrc/jit/fuser/common/iriostream.h>
#include <torch/csrc/jit/fuser/common/iter_visitor.h>
#include <torch/csrc/jit/fuser/common/predicate_compute.h>
#include <torch/csrc/jit/fuser/common/transform_iter.h>

#include <ostream>
#include <stack>
#include <map>

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

struct FindUsedVals : public IterVisitor {
  std::set<Val*> used_vals;

  void handle(Val* v) {
    used_vals.emplace(v);
  }

 public:
  static std::set<Val*> find() {
    FindUsedVals finder;
    finder.traverse(FusionGuard::getCurFusion(), true);
    return finder.used_vals;
  }

};

struct TORCH_API CodeWrite : public IRPrinter {
 private:
  bool isTVOp(const Expr* expr);

  void printIndexInto(std::vector<Int*> indices, const TensorView* const);
  bool print_predicate(const TensorView* const);

  // Print lhs of uop/bop, returns if predicate was needed
  void printAlloc(TensorView*);
  bool printLHS(TensorView*);
  void print(const TensorView* const);
  void print(const Val* const);
  void print(const UnaryOp* const);
  void print(const BinaryOp* const);

  void print(const Split* const) {}
  void print(const Merge* const) {}
  void print(const Reorder* const) {}

  void indent();
  void handle(Expr*);
  void handle(UnaryOp*);
  void handle(BinaryOp*);

  void updateView(TensorView*);

  std::vector<Int*> getLoopIndices();
  void openFor(IterDomain*);
  void closeFor();
  void resetFors();
  void clearActiveView();

  bool producer = false;
  TensorView* consumer = nullptr;

  std::vector<std::pair<Int*, IterDomain*>> fors;
  int indent_size = 0;

  const TensorView* active_view = nullptr;
  int active_view_axis = 0;

  bool reset_fors = false;

  std::set<IterDomain*> bound_iters;
  std::map<const Val* const, std::string> overrides;
  // Grab all values that are used. Look for Tensors
  // to set Int* -> Tensor->size(i)
  // Grab all IterDoms that are used. Look for any
  // mappings to threadIdx.x / blockIdx.x
  // Add them to bound_iters
  void bind(IterDomain* id, Val* iterator);
  void setupOverrides();

  std::map<const Val* const, std::string>::iterator
    overrides_find(Val* val){
      return overrides.find(const_cast<const Val* const>(val));
    }

  void overrides_emplace(Val* val, std::string str){
    overrides[const_cast<const Val* const>(val)] = str;
  }

  void header();

 public:
  CodeWrite(std::ostream& _os) : IRPrinter(_os) {}

  void traverse(
      Fusion* fusion,
      bool from_outputs_only = false,
      bool breadth_first = false,
      std::unordered_set<ValType> val_types = {});
};

} // namespace fuser
} // namespace jit
} // namespace torch