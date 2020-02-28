#pragma once

#include <torch/csrc/jit/fuser/common/iter_visitor.h>
#include <torch/csrc/jit/fuser/common/transform_iter.h>
#include <torch/csrc/jit/fuser/common/arith.h>

#include <torch/csrc/jit/fuser/common/iriostream.h>

#include <ostream>
#include <stack>

namespace torch {
namespace jit {
namespace fuser {

std::ostream& operator<<(std::ostream& os, std::vector<Int*> vec) {
  os << "<";
  for (int i = 0; i < vec.size(); i++) {
    print_inline(os, vec[i]);
    if (i == vec.size() - 1)
      os << ">";
    else
      os << ",";
  }
  return os;
}

struct TORCH_API IndexCompute : public TransformIter {
 protected:
  void replayBackward(Split* expr) override {
    int ax = expr->axis();
    TORCH_CHECK(ax >= 0 && ax + 1 < indices.size());
    indices[ax] = static_cast<Int*>(mul(indices[ax], indices[ax+1]));
    indices.erase(indices.begin() + ax + 1);
  }

  void replayBackward(Merge* expr) override {
    int ax = expr->axis();
    TORCH_CHECK(ax >= 0 && ax < indices.size());

    Int* O = expr->in()->axis(ax+1)->size();
    Int* ind = indices[ax];
    indices[ax] = static_cast<Int*>(div(ind, O));
    indices.insert(indices.begin()+ax+1, static_cast<Int*>(mod(ind, O)));
  }

void replayBackward(Reorder* expr) override {
  // pos2axis[new_pos] = old_pos Generate new axis2pos map
  const std::vector<int>& pos2axis = expr->pos2axis();

  std::vector<Int*> reordered_indices;
  
  //Reverse the map so we can simply push back into reordered_indices
  //axis2pos[old_pos] = new_pos
  std::vector<int> axis2pos(pos2axis.size(), -1);
  
  for (decltype(pos2axis.size()) i = 0; i < pos2axis.size(); i++) {
    int new_pos = i;
    int old_pos = pos2axis[i];
    TORCH_CHECK(new_pos >= 0 && new_pos < indices.size() && old_pos >= 0 && old_pos < indices.size());
    axis2pos[old_pos] = new_pos;
  }
  for (decltype(axis2pos.size()) i = 0; i < axis2pos.size(); i++) {
    int new_pos = axis2pos[i];
    int old_pos = i;
    //reordered_indices[old_pos] = indices[new_pos];
    reordered_indices.push_back(indices[new_pos]);
  }

  indices = reordered_indices;
}

  std::vector<int> axis_map;
  std::vector<Int*> indices;

 public:
  IndexCompute(const TensorView* tv, std::vector<Int*> _indices) {
    indices = _indices;

    TensorDomain* td = tv->domain();

    bool exclude_reduction = td->size() > indices.size();
    TORCH_CHECK(td->size() >= indices.size());

    // Add fake indices on reduction axes if they aren't there
    // just for bookkeeping of split/merge/reorder.
    if(exclude_reduction)
      for(decltype(td->size()) i{0}; i < td->size(); i++)
        if(td->axis(i)->isReduction())
          indices.insert(indices.begin() + i, new Int(-1));
    TensorDomain* root = TransformIter::runBackward(td, true);
    TORCH_CHECK(root->size() == indices.size());
    //Remove indices associated with reduction axes
    if(exclude_reduction){
      for(auto i = root->size() - 1; i >= 0; i--)
        if(root->axis(i)->isReduction())
          indices.erase(indices.begin() + i);
    }
  }
  static std::vector<Int*> computeIndices(const TensorView* tv, std::vector<Int*> _indices){
    IndexCompute ic(tv, _indices);
    return ic.indices;
  }

};

struct TORCH_API CodeWrite : public IterVisitor {
 private:
  bool parse_inline = false;
  bool producer = false;
  TensorView* consumer = nullptr;

  std::ostream& print_indices(std::ostream& os, const std::vector<Int*>&);
  std::ostream& print(std::ostream& os, const TensorView* const);
  std::ostream& print(std::ostream& os, const Val* const);
  std::ostream& print(std::ostream& os, const UnaryOp* const);
  std::ostream& print(std::ostream& os, const BinaryOp* const);

  void indent();
  void handle(UnaryOp*);
  void handle(BinaryOp*);

  void updateView(TensorView*);

  std::vector<Int*> getLoopIndices();
  void openFor(IterDomain*);
  void closeScope();
  void resetFors();
  void clearActiveView();

  std::vector<std::pair<Int*, Int*> > fors;

  const TensorView* active_view = nullptr;
  int active_view_axis = 0;
  bool reset_fors = false;

 public:
  void traverse(
      const Fusion* const fusion,
      bool from_outputs_only = false,
      bool breadth_first = false,
      std::unordered_set<ValType> val_types = {});
};

} // namespace fuser
} // namespace jit
} // namespace torch