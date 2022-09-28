#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_graphviz.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/debug_utils.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

using GroupSet = VectorOfUniqueEntries<SegmentedGroup*>;

} // namespace

std::vector<SegmentedGroup::NeighborGroup> SegmentedGroup::getNeighborGroups() {
  std::vector<NeighborGroup> neighbors;
  for (auto inp : producer_edges) {
    if (inp->val->isFusionOutput()) {
      // Don't fuse across output nodes, would need to find another path.
      continue;
    }
    neighbors.emplace_back(inp->from, inp);
  }
  for (auto out : consumer_edges) {
    if (out->val->isFusionOutput()) {
      // Don't fuse across output nodes, would need to find another path.
      continue;
    }
    neighbors.emplace_back(out->to, out);
  }
  return neighbors;
}

std::vector<SegmentedGroup*> SegmentedGroup::getNeighbors() {
  std::vector<SegmentedGroup*> neighbors;
  auto neighbors_pair = getNeighborGroups();

  std::transform(
      neighbors_pair.begin(),
      neighbors_pair.end(),
      std::back_inserter(neighbors),
      [](auto& neighbor_group) { return neighbor_group.group; });
  return neighbors;
}

std::vector<SegmentedGroup::NeighborGroup> SegmentedGroup::
    getMergeCandidates() {
  // Don't look for candidates if already merged
  if (merged_) {
    return {};
  }

  std::vector<NeighborGroup> neighbors = getNeighborGroups();

  // Can this node be merged with another? Check if neighbors are merged, if
  // so and merged neighbor is within 1 level or node merged with neighbor is
  // within 1 level, can't merge this node with anything else.
  bool can_merge_this = true;
  for (auto& neighbor : neighbors) {
    if (!neighbor.group->merged_) {
      continue;
    }
    if (std::abs(neighbor.group->level_ - level_) <= 1) {
      can_merge_this = false;
    }
    if (std::abs(neighbor.group->merge_with_->level_ - level_) <= 1) {
      can_merge_this = false;
    }
  }
  if (!can_merge_this) {
    return {};
  }

  std::vector<bool> can_merge(neighbors.size(), true);

  // Find neighbors with a level that is only 1 differant than this groups level
  for (const auto i : c10::irange(neighbors.size())) {
    if (std::abs(neighbors[i].group->level_ - level_) > 1) {
      can_merge[i] = false;
    }
  }

  // Check neighbor of neighbors we're considering, if any of them are merged
  // with another node, make sure the resulting edge wouldn't have a level
  // difference of 1
  for (const auto i : c10::irange(neighbors.size())) {
    if (!can_merge[i]) {
      continue;
    }

    for (auto neighbor_neighbor : neighbors[i].group->getNeighbors()) {
      // Don't check self
      if (neighbor_neighbor == neighbors[i].group) {
        continue;
      }
      if (neighbor_neighbor->merged_) {
        // check neighbor_neighbor level
        if (std::abs(neighbor_neighbor->level_ - level_) <= 1) {
          can_merge[i] = false;
        }
        if (std::abs(neighbor_neighbor->level_ - neighbors[i].group->level_) <=
            1) {
          can_merge[i] = false;
        }

        // check neighbor_neighber->merged_->level_
        if (std::abs(neighbor_neighbor->merge_with_->level_ - level_) <= 1) {
          can_merge[i] = false;
        }
        if (std::abs(
                neighbor_neighbor->merge_with_->level_ -
                neighbors[i].group->level_) <= 1) {
          can_merge[i] = false;
        }
      }
    }
  }

  std::vector<NeighborGroup> merge_candidates;
  for (const auto i : c10::irange(neighbors.size())) {
    if (can_merge[i]) {
      merge_candidates.push_back(neighbors[i]);
    }
  }
  return merge_candidates;
}

void SegmentedGroup::clearTraversalInfo() {
  level_ = -1;
  visited_ = false;
  merge_with_ = nullptr;
  merge_through_ = nullptr;
  merged_ = false;
}

std::vector<Val*> SegmentedGroup::edgesToVals(
    const std::vector<SegmentedEdge*>& se_v) {
  std::vector<Val*> ret_v;
  ret_v.reserve(se_v.size());

  std::transform(
      se_v.cbegin(),
      se_v.cend(),
      std::back_inserter(ret_v),
      [](SegmentedEdge* se) { return se->val; });
  return ret_v;
}

template <typename PREDICATE>
void insertUniquePredicated(
    std::vector<Val*>& v,
    const std::vector<SegmentedEdge*>& e,
    PREDICATE pred) {
  VectorOfUniqueEntries<Val*> to_add;
  for (auto edge : e) {
    to_add.pushBack(edge->val);
  }

  std::copy_if(
      to_add.vector().begin(),
      to_add.vector().end(),
      std::back_inserter(v),
      [pred](Val* val) { return pred(val); });
}

void SegmentedGroup::finalize() {
  // Move all the edges to group input/output
  // Inputs
  insertUniquePredicated(
      input_vals, producer_edges, [](Val* v) { return !v->isFusionInput(); });

  std::unordered_set<Val*> input_set(input_vals.begin(), input_vals.end());

  for (auto expr : exprs_) {
    for (auto i : expr->inputs()) {
      if (i->isAnInt() && i->definition() == nullptr && !i->isConstScalar() &&
          !i->isFusionInput() && !input_set.count(i)) {
        input_set.insert(i);
        input_vals.push_back(i);
      }
    }
  }

  // Outputs
  insertUniquePredicated(
      output_vals, consumer_edges, [](Val* v) { return !v->isFusionOutput(); });

  // alias aware segmentation. we add inputs that are aliased by output
  // generated in this SegmentedGroup
  for (auto output : output_vals) {
    if (auto aliased_input = segmented_fusion_->findAlias(output)) {
      // aliasing currently only supported as output to input
      TORCH_INTERNAL_ASSERT(
          aliased_input->isFusionInput(),
          "aliased input is not found in the complete fusion");
      if (!input_set.count(aliased_input)) {
        input_set.insert(aliased_input);
        input_vals.push_back(aliased_input);
      }
    }
  }
}

std::ostream& operator<<(std::ostream& os, const SegmentedGroup* group) {
  os << "g{";
  auto expr_to_print = group->exprs();
  std::sort(
      expr_to_print.begin(),
      expr_to_print.end(),
      [](auto expr_a, auto expr_b) -> bool {
        return expr_a->name() < expr_b->name();
      });
  for (const auto i : c10::irange(expr_to_print.size())) {
    os << expr_to_print[i]->name();
    if (i + 1 != expr_to_print.size())
      os << ", ";
  }
  os << "}\n";
  return os;
}

void SegmentedGroup::print() const {
  std::cout << this << "\n";
}

std::string toString(const SegmentedGroup* group) {
  std::stringstream ss;
  ss << group;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const SegmentedEdge* edge) {
  os << "e{ " << edge->from << " -> " << edge->to << "(";
  IrPrinter irp(os);
  irp.handle(edge->val);
  os << ") }\n";
  return os;
}

void SegmentedEdge::print() const {
  std::cout << this << "\n";
}

std::string toString(const SegmentedEdge* edge) {
  std::stringstream ss;
  ss << edge;
  return ss.str();
}

std::unique_ptr<SegmentedFusion> SegmentedFusion::fromCompleteFusion(
    std::unique_ptr<Fusion> fusion_ptr,
    ScheduleHeuristic heuristic) {
  auto fusion = fusion_ptr.get();

  auto segmented_fusion_ptr =
      std::make_unique<SegmentedFusion>(std::move(fusion_ptr));

  // Make a group for the single fusion
  auto single_group = segmented_fusion_ptr->newGroup();

  // Add input and output vals
  single_group->input_vals = fusion->inputs();
  single_group->output_vals = fusion->outputs();

  // Get ordered expression list
  single_group->resetExprList();

  // Assign heuristics and id for the complete fusion
  //  to share the runtime path of segmented fusion.
  single_group->setHeuristic(heuristic);
  single_group->setID(0);

  return segmented_fusion_ptr;
}

SegmentedFusion::SegmentedFusion(std::unique_ptr<Fusion> fusion)
    : impl_(this), complete_fusion_(std::move(fusion)) {
  segmented_fusion_name_ = segmentedFusionName();
  annotateFP16IntermediateTensors();
}

SegmentedGroup* SegmentedFusion::Impl::makeGroup() {
  groups_.emplace_back(std::make_unique<SegmentedGroup>(owning_fusion_));
  return groups_.back().get();
}

SegmentedGroup* SegmentedFusion::Impl::makeGroup(Expr* expr) {
  groups_.emplace_back(std::make_unique<SegmentedGroup>(expr, owning_fusion_));
  return groups_.back().get();
}

SegmentedEdge* SegmentedFusion::Impl::makeEdge(
    SegmentedGroup* from,
    SegmentedGroup* to,
    Val* val) {
  edges_.emplace_back(std::make_unique<SegmentedEdge>(from, to, val));
  return edges_.back().get();
}

void SegmentedFusion::Impl::cleanUnused() {
  std::unordered_set<SegmentedGroup*> g_used(
      owning_fusion_->groups().begin(), owning_fusion_->groups().end());
  std::unordered_set<SegmentedEdge*> e_used(
      owning_fusion_->edges().begin(), owning_fusion_->edges().end());

  groups_.erase(
      std::remove_if(
          groups_.begin(),
          groups_.end(),
          [&g_used](auto& g) { return g_used.count(g.get()) == 0; }),
      groups_.end());

  edges_.erase(
      std::remove_if(
          edges_.begin(),
          edges_.end(),
          [&e_used](auto& e) { return e_used.count(e.get()) == 0; }),
      edges_.end());
}

SegmentedGroup* SegmentedFusion::newGroup() {
  SegmentedGroup* g = impl_.makeGroup();
  groups_.push_back(g);
  return g;
}

SegmentedGroup* SegmentedFusion::newGroup(Expr* expr) {
  SegmentedGroup* g = impl_.makeGroup(expr);
  groups_.push_back(g);
  return g;
}

SegmentedEdge* SegmentedFusion::newEdge(
    SegmentedGroup* from,
    SegmentedGroup* to,
    Val* val) {
  SegmentedEdge* e = impl_.makeEdge(from, to, val);
  edges_.push_back(e);
  return e;
}

void SegmentedFusion::draw() {
  size_t group_index = 0;
  std::unordered_map<const Expr*, size_t> expr_color_map;

  for (auto group : groups()) {
    for (auto expr : group->exprs()) {
      if (ir_utils::isTvOp(expr)) {
        expr_color_map[expr] = group_index;
      }
    }
    group_index++;
  }

  std::stringstream sstream;
  sstream << "segmented_fusion" << segmented_fusion_name_ << ".dot";
  auto filename = sstream.str();

  IrGraphGenerator::print(
      completeFusion(),
      filename.c_str(),
      IrGraphGenerator::DetailLevel::ComputeOnly,
      &expr_color_map);
}

namespace {

std::vector<Val*> uniqueValConcat(
    const std::vector<std::vector<Val*>>& val_vecs) {
  std::vector<Val*> unique_vals;
  std::unordered_set<Val*> added;
  for (const auto& vec : val_vecs) {
    for (auto val : vec) {
      if (added.find(val) == added.end()) {
        unique_vals.push_back(val);
        added.emplace(val);
      }
    }
  }
  return unique_vals;
}

// Concat's producer edges of sg1 and sg2, but removes any edges from/to sg1/sg2
std::vector<SegmentedEdge*> getMergedProducerEdges(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2) {
  TORCH_INTERNAL_ASSERT(
      sg1 != nullptr && sg2 != nullptr,
      "This function doesn't handle trivial.");

  auto producer_edges = sg1->producer_edges;

  producer_edges.insert(
      producer_edges.end(),
      sg2->producer_edges.begin(),
      sg2->producer_edges.end());

  // Register producers into sg2
  std::unordered_set<Val*> sg2_vals;
  for (auto se : sg2->producer_edges) {
    sg2_vals.emplace(se->val);
  }

  producer_edges.erase(
      std::remove_if(
          producer_edges.begin(),
          producer_edges.end(),
          [&sg1, &sg2, &sg2_vals](SegmentedEdge* se) {
            // remove edges in between the groups and common uses
            return (se->to == sg1 && se->from == sg2) ||
                (se->to == sg2 && se->from == sg1) ||
                (se->to == sg1 && sg2_vals.count(se->val));
          }),
      producer_edges.end());

  // Remove Duplicate Edges

  return producer_edges;
}

// Concat's consumer edges of sg1 and sg2, but removes any edges from/to sg1/sg2
std::vector<SegmentedEdge*> getMergedConsumerEdges(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2) {
  TORCH_INTERNAL_ASSERT(
      sg1 != nullptr && sg2 != nullptr,
      "This function doesn't handle trivial.");

  auto consumer_edges = sg1->consumer_edges;
  consumer_edges.insert(
      consumer_edges.end(),
      sg2->consumer_edges.begin(),
      sg2->consumer_edges.end());

  consumer_edges.erase(
      std::remove_if(
          consumer_edges.begin(),
          consumer_edges.end(),
          [&sg1, &sg2](SegmentedEdge* se) {
            return (se->to == sg1 && se->from == sg2) ||
                (se->to == sg2 && se->from == sg1);
          }),
      consumer_edges.end());

  return consumer_edges;
}

// Returns a determinstic, unique set of inputs of the segment group, sg1, or
// the combined group sg1 + sg2
std::vector<Val*> getAllInputs(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2 = nullptr) {
  std::vector<SegmentedEdge*> merged_producer_edges;

  if (sg1 != nullptr && sg2 != nullptr) {
    merged_producer_edges = getMergedProducerEdges(sg1, sg2);
  } else if (sg1 != nullptr) {
    merged_producer_edges = sg1->producer_edges;
  } else if (sg2 != nullptr) {
    merged_producer_edges = sg2->producer_edges;
  }

  std::vector<Val*> producer_edge_vals;

  std::transform(
      merged_producer_edges.begin(),
      merged_producer_edges.end(),
      std::back_inserter(producer_edge_vals),
      [](SegmentedEdge* se) { return se->val; });

  return uniqueValConcat(
      {sg1 == nullptr ? std::vector<Val*>() : sg1->input_vals,
       sg2 == nullptr ? std::vector<Val*>() : sg2->input_vals,
       producer_edge_vals});
}

// Returns a determinstic, unique set of outputs of the segment group, sg1, or
// the combined group sg1 + sg2
std::vector<Val*> getAllOutputs(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2 = nullptr) {
  std::vector<SegmentedEdge*> merged_consumer_edges;

  if (sg1 != nullptr && sg2 != nullptr) {
    merged_consumer_edges = getMergedConsumerEdges(sg1, sg2);
  } else if (sg1 != nullptr) {
    merged_consumer_edges = sg1->consumer_edges;
  } else if (sg2 != nullptr) {
    merged_consumer_edges = sg2->consumer_edges;
  }

  std::vector<Val*> consumer_edge_vals;

  std::transform(
      merged_consumer_edges.begin(),
      merged_consumer_edges.end(),
      std::back_inserter(consumer_edge_vals),
      [](SegmentedEdge* se) { return se->val; });

  auto output_vals = uniqueValConcat(
      {sg1 == nullptr ? std::vector<Val*>() : sg1->output_vals,
       sg2 == nullptr ? std::vector<Val*>() : sg2->output_vals,
       consumer_edge_vals});

  return output_vals;
}

// Set version of getting merged input or output if segmented_groups were
//  merged
//  outputs respects order in segmented_groups for deterministic
//  merge trace
//  will get input if get_inputs otherwise will get ouputs
//  TODO: merge with the binary counter parts
std::vector<Val*> allInputsIfTrueElseOutputs(
    const std::vector<SegmentedGroup*>& segmented_groups,
    bool get_inputs = true) {
  // Helper to distinguish if we are getting inputs or outputs
  using EdgeVec = std::vector<SegmentedEdge*>;
  using ValVec = std::vector<Val*>;

  // Get producer edges to get inputs, consumer edges to get outputs
  auto edges_to_process_from_or_to_group =
      [get_inputs](SegmentedGroup* group) -> EdgeVec& {
    return get_inputs ? group->producer_edges : group->consumer_edges;
  };

  // Get the group that is connected to current group
  auto global_vals_from_or_to_group =
      [get_inputs](SegmentedGroup* group) -> ValVec& {
    return get_inputs ? group->input_vals : group->output_vals;
  };

  // Get the group that is connected to current group by given edge
  auto opposite_end_of_edge = [get_inputs](SegmentedEdge* edge) {
    return get_inputs ? edge->from : edge->to;
  };

  // Keep track of value and order to ensure deterministic result
  std::vector<Val*> merged_vals;
  std::unordered_set<Val*> merged_vals_set;

  // Put groups in a set for quick look up
  std::unordered_set<SegmentedGroup*> segmented_groups_set(
      segmented_groups.begin(), segmented_groups.end());

  // Collect vals associated with edges
  for (auto group : segmented_groups) {
    for (auto edge : edges_to_process_from_or_to_group(group)) {
      if (
          // Need to de-duplicate values so we don't get multiple of any input
          !merged_vals_set.count(edge->val) &&
          // One side of this edge will be `group`, if the other end is
          //  also in segmented_groups, then this is an internal edge
          //  that we don't want.
          !segmented_groups_set.count(opposite_end_of_edge(edge))) {
        merged_vals.push_back(edge->val);
        merged_vals_set.insert(edge->val);
      }
    }
  }

  // Collect original fusion's inputs/outputs and append at the end
  for (auto group : segmented_groups) {
    for (auto global_val : global_vals_from_or_to_group(group)) {
      // de-duplicate
      if (!merged_vals_set.count(global_val)) {
        merged_vals.push_back(global_val);
        merged_vals_set.insert(global_val);
      }
    }
  }

  return merged_vals;
}

// A sorting utility used for debug printing only
//  sorts the given vector of expressions in topological
//  order, with equal cases respecting the original order
//  in the vector.
std::vector<Expr*> groupExprPrintSorting(const std::vector<Expr*>& exprs) {
  std::vector<Expr*> exprs_to_print(exprs.begin(), exprs.end());
  std::unordered_set<Expr*> exprs_to_print_set(exprs.begin(), exprs.end());
  std::unordered_set<Expr*> exprs_visited;
  std::vector<Expr*> sorted_list;
  while (!std::all_of(
      exprs_to_print.begin(),
      exprs_to_print.end(),
      [&exprs_visited](auto expr) { return exprs_visited.count(expr); })) {
    bool expr_added_to_sorted_list = false;
    for (auto expr : exprs_to_print) {
      if (!exprs_visited.count(expr)) {
        bool add_this_expr = true;
        // Check if any of the inputs of current
        //  expression within the group
        //  hasn't been visited
        for (auto input : expr->inputs()) {
          if (input->definition() &&
              exprs_to_print_set.count(input->definition()) &&
              !exprs_visited.count(input->definition())) {
            add_this_expr = false;
            break;
          }
        }

        // Append the current group to sorted list
        //  and mark visited
        if (add_this_expr) {
          expr_added_to_sorted_list = true;
          exprs_visited.insert(expr);
          sorted_list.push_back(expr);
          break;
        }
      }
    }
    TORCH_INTERNAL_ASSERT(
        expr_added_to_sorted_list,
        "group debug print failed, exprs within given vector not a DAG");
  }
  return sorted_list;
}

// Utility function to list all expressions in a group
void detailGroupPrint(std::ostream& os, const SegmentedGroup* group) {
  IrPrinter irp(os);

  auto sort_val_by_name = [](std::vector<Val*> vals_to_sort) {
    std::sort(vals_to_sort.begin(), vals_to_sort.end(), [](Val* a, Val* b) {
      return a->name() < b->name();
    });
    return vals_to_sort;
  };

  os << "g{"
     << "(" << toString(group->heuristic()) << ")\n";
  os << "inputs: \n";
  for (auto input : sort_val_by_name(getAllInputs(group))) {
    os << input << " " << input->getDataType().value() << "\n";
  }
  os << "outputs: \n";
  for (auto output : sort_val_by_name(getAllOutputs(group))) {
    os << output << " " << output->getDataType().value() << "\n";
  }

  os << "\n\n";

  auto expr_to_print = groupExprPrintSorting(group->exprs());

  for (const auto i : c10::irange(expr_to_print.size())) {
    irp.handle(expr_to_print[i]);
  }
  os << "}\n\n";
}

//! Insert casts for an intermediate tensorview, i.e. ones
//!  that are in segmentedEdges. The insertion is done on
//!  the complete fusion, which should be owned by a segmented
//!  fusion so that only one segmented fusion will be affected.
//!  The replacement pattern is:
//!                 TV0
//!     replaced as:
//!       fp16_tv = cast(TV0)
//!       fp32_tv = cast(fp16_tv)
//!
//!  All segmented groups that take TV0 as input will then
//!   take fp16_tv or bf16_tv instead and the cast to fp32 will be
//!   automatically included in each of the groups.
TensorView* castIntermediateValueInCompleteFusion(
    Fusion* fusion,
    TensorView* original_tv,
    std::unordered_set<Expr*> edge_from_group_uses,
    DataType dtype) {
  FusionGuard fg(fusion);

  // A utility lambda that creates consumer tensordomain of
  //  the given tv and create a new tensorview around the
  //  new tensordomain with the given data type.
  auto make_consumer_tv = [&](TensorView* from, DataType data_type) {
    // Keep broadcast axes and remove reduction axes
    size_t i = 0;
    auto no_reduction_root_domain =
        TensorDomain::noReductions(original_tv->getMaybeRFactorDomain());
    std::vector<IterDomain*> new_root_domain(no_reduction_root_domain.size());
    for (const auto& dom : no_reduction_root_domain) {
      new_root_domain[i++] = dom->cloneWithoutRFactor();
    }

    // Create the actual domain and tv.
    return IrBuilder::create<TensorView>(
        IrBuilder::create<TensorDomain>(
            new_root_domain, std::vector<bool>(new_root_domain.size(), true)),
        data_type);
  };

  // create the tv's to cast
  auto half_precision_tv = make_consumer_tv(original_tv, dtype);

  auto fp32_tv = make_consumer_tv(original_tv, DataType::Float);

  // replace uses of original tv with fp32_tv in the complete
  //  fusion
  for (auto expr : fusion->unordered_uses(original_tv)) {
    // Don't modify internal uses of buffers, only cast for outputs.
    if (edge_from_group_uses.find(expr) == edge_from_group_uses.end()) {
      ir_utils::replaceValInExpr(expr, original_tv, fp32_tv);
    }
  }

  // Insert the cast ops.
  IrBuilder::create<UnaryOp>(UnaryOpType::Cast, half_precision_tv, original_tv);
  IrBuilder::create<UnaryOp>(UnaryOpType::Cast, fp32_tv, half_precision_tv);

  // Return the new tv to replace original tv with
  //  on the segmented edges.
  return half_precision_tv;
}
} // namespace

void SegmentedFusion::finalize() {
  impl_.cleanUnused();
  // Insert casts for the tensorviews that are on
  //  segmented edges and also on the force_to_fp16 list
  //
  // Note:
  //  The cast is inserted after the segmenter canSchedule check, which
  //  shouldn't cause problem short-term. The reason we put the cast here
  //  is  we don't want to keep making copies of the original fusion
  //  during segmentation. Could consider making the cast insertion
  //  reversible if we do have to test canSchedule with the casts inserted
  //  during segmentation process in the future.

  // Keep track of groups that need to update expr list,
  //  including both the producer and consumer of the selected tv's that
  //  we cast to fp16.
  std::unordered_set<SegmentedGroup*> affected_group_set;
  // A map to keep track of the tv's that have been inserted cast
  //  and its fp16 version.
  std::unordered_map<TensorView*, TensorView*> fp32_to_half_cast_map;

  // Go through all edges of the segmented fusion.
  for (auto edge : edges()) {
    TORCH_INTERNAL_ASSERT(edge->val->isA<TensorView>());
    auto edge_tv = edge->val->as<TensorView>();

    // Uses of the edge value within the from group should not be replaced. This
    // will cause the group to have an intermediate tensor
    // tv -> float2half -> output
    //            \ -> half2float -> other uses in group
    // The conversion back and forth from half precision can hurt numerics.
    // Collect expressions that use the edge value of concern within the from
    // group to avoid replacing with the cast tensor.
    std::unordered_set<Expr*> uses_in_from_group;

    // All expressions in the from group of the edge
    std::unordered_set<Expr*> from_group_exprs(
        edge->from->exprs().begin(), edge->from->exprs().end());

    // All uses of the edge val
    for (auto edge_val_use_expr : edge_tv->uses()) {
      if (from_group_exprs.count(edge_val_use_expr)) {
        // Find uses in the to group of the val
        uses_in_from_group.emplace(edge_val_use_expr);
      }
    }

    // Only look at ones that need to cast to fp16 or bf16
    if ((force_fp16_tv_set_.count(edge_tv) > 0)) {
      auto cast_tv_it = fp32_to_half_cast_map.find(edge->val->as<TensorView>());
      TensorView* cast_tv = nullptr;
      // Insert cast ops for this tv if we haven't done so.
      if (cast_tv_it == fp32_to_half_cast_map.end()) {
        cast_tv = castIntermediateValueInCompleteFusion(
            complete_fusion_.get(),
            edge_tv,
            uses_in_from_group,
            force_half_precision_type_);
        fp32_to_half_cast_map[edge->val->as<TensorView>()] = cast_tv;
      } else {
        cast_tv = cast_tv_it->second;
      }

      // Update the edge to use the fp16 version
      edge->val = cast_tv;

      // Mark the groups for update later
      affected_group_set.insert(edge->from);
      affected_group_set.insert(edge->to);

      // The expr pointers on the group's expr list might have been freed
      //  by now after `ir_utils::replaceValInExpr`.
      // Need a valid expression list to continue. Update from and to group.
      edge->from->resetExprList();
      edge->to->resetExprList();
    }
  }
}

//! An utility class to compute and maintain the "producers of"
//!   relationship in a segmented graph. Space heavy and should
//!   avoid use on very large graphs.
//!
//!  Currently trying to move as far as possible with only a
//!   producer map, without transposing it to make a consumer map.
//!  Making it NonCopyable because we should never need to
//!   copy an instance of this class.
//!  TODO: Space efficiency of this class will be important,
//!        because we need it in the pre-merging of segmentedGroups,
//!        currently O(n^2). O(nlogn) would be a reasonable
//!        goal to achieve.
class GroupDependencyAnalysis : public NonCopyable, public SegmenterAnalysis {
  using GroupSetOwningPtr = std::unique_ptr<GroupSet>;
  using DependencyMap = std::unordered_map<SegmentedGroup*, GroupSetOwningPtr>;

 public:
  //! Populate producers of all groups in segmented fusion
  explicit GroupDependencyAnalysis(const SegmentedFusion* segmented_fusion)
      : segmented_fusion_(segmented_fusion) {
    computeAllProducers();
  }

  //! Checks if group is consumer of any group in groups_to_check
  //!  TODO: refactor this similar to isConsumerOf
  bool isConsumerOfAny(
      SegmentedGroup* group,
      const std::vector<SegmentedGroup*>& groups_to_check) {
    auto& producers_of_group = getAllKnownProducersSet(group);
    for (const auto& potential_producer : groups_to_check) {
      if (producers_of_group->has(potential_producer)) {
        return true;
      }
    }
    return false;
  }

  bool isConsumerOf(SegmentedGroup* a, SegmentedGroup* b) {
    auto it = known_producers_of_.find(a);
    if (it == known_producers_of_.end()) {
      return false;
    }
    return it->second->has(b);
  }

  bool isProducerOf(SegmentedGroup* a, SegmentedGroup* b) {
    return isConsumerOf(b, a);
  }

  //! Finds the common producers of given set of groups
  GroupSet getCommonProducersOf(std::vector<SegmentedGroup*> groups);

  //! Update the map when the given two groups have been merged to create `ab`
  //! this method is for book keeping and query only, doesn't implicitly check
  //!  for DAG
  void mergeGroups(SegmentedGroup* a, SegmentedGroup* b, SegmentedGroup* ab);

  //! Update the map when the given two groups have been merged to create
  //! `merged` this method is for book keeping and query only, doesn't
  //! implicitly check
  //!  for DAG
  void mergeGroups(const GroupSet& groups, SegmentedGroup* merged);

  //! Populate all values that is on a path from producer to consumer
  //!  efficiency can be important here. (TODO)
  GroupSet valuesBetween(SegmentedGroup* producer, SegmentedGroup* consumer) {
    if (producer == consumer) {
      return {};
    }

    GroupSet values_between;
    auto& all_producers_of_consumer = known_producers_of_.at(consumer);
    TORCH_INTERNAL_ASSERT(
        all_producers_of_consumer->has(producer),
        "Fusion segment: Trying to compute path between two nodes that are not producer-consumer pairs");

    for (auto producer_of_consumer : *all_producers_of_consumer) {
      if (known_producers_of_.at(producer_of_consumer)->has(producer)) {
        values_between.pushBack(producer_of_consumer);
      }
    }

    return values_between;
  }

  //! Checks if the segmented fusion this class tracks is still a DAG
  //!  used for generating assertions after transforms
  bool isproducerMapDAG() const {
    for (auto& it : known_producers_of_) {
      if (it.second->has(it.first)) {
        return false;
      }
    }
    return true;
  }

 private:
  //! Collect initial producer info using
  //!  a work list algorithm through forward traversal
  //!  a backward DFS would do the same
  void computeAllProducers();

  //! Add all consumers of `producer` to `to_visit`
  void addConsumersToWorkList(SegmentedGroup* producer, GroupSet& to_visit) {
    for (auto e : producer->consumer_edges) {
      // A consumer wouldn't have been worked before any of its producer
      to_visit.pushBack(e->to);
    }
  }

  //! Propagate all known producers of `from` into `into`, used to keep track
  //! of:
  //!  1. `from` is a producer of `into`
  //!  2. `from` has been merged with other group to create `into`
  void mergeAllKnownProducersIntoFrom(
      SegmentedGroup* into,
      SegmentedGroup* from) {
    auto& producer_set_to_merge = *getAllKnownProducersSet(from);
    for (auto group : producer_set_to_merge) {
      getAllKnownProducersSet(into)->pushBack(group);
    }
  }

  //! Utility to access known producers of a group so far
  GroupSetOwningPtr& getAllKnownProducersSet(SegmentedGroup* group) {
    auto& producer_set_ptr = known_producers_of_[group];
    if (!producer_set_ptr) {
      producer_set_ptr = std::make_unique<GroupSet>();
    }
    return producer_set_ptr;
  }

  // utility to compute the set intersection of group sets a,b
  GroupSet groupSetIntersection(const GroupSet& a, const GroupSet& b) {
    bool a_is_smaller = a.size() < b.size();
    const auto& smaller_group_set = a_is_smaller ? a : b;
    const auto& bigger_group_set = a_is_smaller ? b : a;

    GroupSet intersection;
    for (auto group : smaller_group_set) {
      if (bigger_group_set.has(group)) {
        intersection.pushBack(group);
      }
    }
    return intersection;
  }

 private:
  const SegmentedFusion* segmented_fusion_;
  DependencyMap known_producers_of_;
};

//! Finds the common producers of given set of groups
GroupSet GroupDependencyAnalysis::getCommonProducersOf(
    std::vector<SegmentedGroup*> groups) {
  if (groups.empty()) {
    return {};
  }

  // Optimization: start with the smallest producer set
  std::sort(
      groups.begin(),
      groups.end(),
      [this](SegmentedGroup* a, SegmentedGroup* b) {
        return known_producers_of_.at(a)->size() <
            known_producers_of_.at(b)->size();
      });

  // Get intersection of producers
  GroupSet common_producers = *(known_producers_of_.at(groups[0]));
  for (const auto i : c10::irange(1, groups.size())) {
    common_producers = groupSetIntersection(
        common_producers, *(known_producers_of_.at(groups[i])));
  }

  return common_producers;
}

//! Update the map when the given two groups have been merged to create `ab`
//! this method is for book keeping and query only, doesn't implicitly check
//!  for DAG
void GroupDependencyAnalysis::mergeGroups(
    SegmentedGroup* a,
    SegmentedGroup* b,
    SegmentedGroup* ab) {
  // Access/Create the producer set of ab
  auto& ab_set = getAllKnownProducersSet(ab);

  // propagate a's and b's known producers into ab
  mergeAllKnownProducersIntoFrom(ab, a);
  mergeAllKnownProducersIntoFrom(ab, b);

  // a, b are now merged, so no longer exist
  ab_set->erase(a);
  ab_set->erase(b);

  // a, b no longer exist, remove their producer sets
  known_producers_of_.erase(a);
  known_producers_of_.erase(b);

  // update producer maps of other groups
  for (auto& it : known_producers_of_) {
    // for all groups that are produced by either a or b
    if (it.second->has(a) || it.second->has(b)) {
      // insert ab as the new producer
      it.second->pushBack(ab);
      // all producers of both a and b are now producers of `it`
      mergeAllKnownProducersIntoFrom(it.first, ab);
    }
    // a, b no longer exist, remove them from `it`
    it.second->erase(a);
    it.second->erase(b);
  }
}

//! Update the map when the given two groups have been merged to create
//! `merged` this method is for book keeping and query only, doesn't
//! implicitly check
//!  for DAG
void GroupDependencyAnalysis::mergeGroups(
    const GroupSet& groups,
    SegmentedGroup* merged) {
  // Access/Create the producer set of merged
  auto& merged_set = getAllKnownProducersSet(merged);

  // Populate all producers of groups and
  //  write into producer map of merged
  std::for_each(
      groups.begin(), groups.end(), [this, merged](SegmentedGroup* group) {
        mergeAllKnownProducersIntoFrom(merged, group);
      });

  // Erase all groups that was merged from producer map
  std::for_each(
      groups.begin(), groups.end(), [this, &merged_set](SegmentedGroup* group) {
        // erase inter dependencies
        merged_set->erase(group);
        // erase producer map tracking merged entires
        known_producers_of_.erase(group);
      });

  // Update producer relationships with other groups in producer map
  for (auto& it : known_producers_of_) {
    auto producer_intersection = groupSetIntersection(*(it.second), groups);
    // if current node has any producer that was merged
    if (producer_intersection.size() > 0) {
      for (auto merged_producer : producer_intersection) {
        // delete all disappearing producers
        it.second->erase(merged_producer);
      }
      // insert the new group as producer
      it.second->pushBack(merged);
    }
  }
}

//! Collect initial producer info using
//!  a work list algorithm through forward traversal
//!  a backward DFS would do the same
void GroupDependencyAnalysis::computeAllProducers() {
  GroupSet visited;
  GroupSet to_visit;

  // Collect source nodes, with no producers we are guaranteed
  //  a source node on a DAG
  for (auto group : segmented_fusion_->cgroups()) {
    if (group->producer_edges.empty()) {
      visited.pushBack(group);
    }
  }

  // visited now only contain source nodes
  //  they can go backward to nowhere
  for (auto group : visited) {
    addConsumersToWorkList(group, to_visit);
  }

  while (!to_visit.empty()) {
    SegmentedGroup* to_update = nullptr;
    for (auto visiting_group : to_visit) {
      if (std::all_of(
              visiting_group->producer_edges.begin(),
              visiting_group->producer_edges.end(),
              [&visited](SegmentedEdge* e) { return visited.has(e->from); })) {
        // filter multi-edges
        GroupSet producers_of_visiting_group;
        for (auto edge : visiting_group->producer_edges) {
          producers_of_visiting_group.pushBack(edge->from);
        }

        // populate all possible paths
        // from producer backward, including
        // the producer
        for (auto producer : producers_of_visiting_group) {
          getAllKnownProducersSet(visiting_group)->pushBack(producer);
          mergeAllKnownProducersIntoFrom(visiting_group, producer);
        }
        to_update = visiting_group;
        break;
      }
    }
    if (to_update) {
      addConsumersToWorkList(to_update, to_visit);
      to_visit.erase(to_update);
      visited.pushBack(to_update);
    } else {
      TORCH_INTERNAL_ASSERT(false, "unreachable, original graph not a DAG");
    }
  }
}

std::ostream& operator<<(
    std::ostream& os,
    const SegmentedFusion* segmented_fusion) {
  // Topologically sort groups
  GroupDependencyAnalysis dependency(segmented_fusion);
  std::vector<SegmentedGroup*> groups_to_print(
      segmented_fusion->cgroups().begin(), segmented_fusion->cgroups().end());
  std::vector<SegmentedGroup*> sorted_groups_to_print;

  // Sort groups topologically from producer to consumer before printing
  while (!groups_to_print.empty()) {
    auto group_it_to_append = groups_to_print.begin();
    for (auto group_it_to_compare = groups_to_print.begin();
         group_it_to_compare != groups_to_print.end();
         group_it_to_compare++) {
      if (dependency.isProducerOf(*group_it_to_compare, *group_it_to_append)) {
        group_it_to_append = group_it_to_compare;
      }
    }
    sorted_groups_to_print.push_back(*group_it_to_append);
    groups_to_print.erase(group_it_to_append);
  }

  // Do a reverse look up to check the order of sorted groups
  std::unordered_map<SegmentedGroup*, size_t> group_order;
  for (const auto i : c10::irange(sorted_groups_to_print.size())) {
    group_order[sorted_groups_to_print[i]] = i;
  }

  // Sort edges to print
  std::vector<SegmentedEdge*> sorted_edges_to_print(
      segmented_fusion->cedges().begin(), segmented_fusion->cedges().end());
  std::sort(
      sorted_edges_to_print.begin(),
      sorted_edges_to_print.end(),
      [&group_order](SegmentedEdge* edge_a, SegmentedEdge* edge_b) {
        return group_order.at(edge_a->from) < group_order.at(edge_b->from);
      });

  os << "Segmented_Fusion Dump: -- fusion segments:\n";
  os << "Segmented_Fusion{ \n";
  os << "groups: \n";
  for (const auto g : sorted_groups_to_print) {
    os << g << "\n";
  }
  os << "edges: \n";
  for (const auto e : sorted_edges_to_print) {
    os << e << "\n";
  }
  os << "\ngroup details:\n";
  for (const auto g : sorted_groups_to_print) {
    detailGroupPrint(os, g);
  }
  os << "} //Segmented_Fusion\n";
  return os;
}

void SegmentedFusion::print() const {
  std::cout << "Segmented_Fusion Dump: -- Re-written complete fusion:{\n";
  completeFusion()->printMath();
  std::cout << "} // {Re-written complete fusion}\n";
  std::cout << this << "\n";
}

std::string toString(SegmentedFusion* segmented_fusion) {
  std::stringstream ss;
  ss << segmented_fusion;
  return ss.str();
}

std::unique_ptr<Fusion> SegmentedFusion::makeFusion(SegmentedGroup* sg) {
  std::unique_ptr<Fusion> fusion_segment = std::make_unique<Fusion>();

  auto complete_to_segment_map =
      Fusion::copy(completeFusion(), fusion_segment.get());

  std::vector<Val*> input_list(
      fusion_segment->inputs().begin(), fusion_segment->inputs().end());
  for (auto inp : input_list) {
    fusion_segment->removeInput(inp);
  }

  std::vector<Val*> output_list(
      fusion_segment->outputs().begin(), fusion_segment->outputs().end());
  for (auto out : output_list) {
    fusion_segment->removeOutput(out);
  }

  std::vector<TensorView*> view_tvs;
  for (auto inp : getAllInputs(sg)) {
    auto clone_tv = complete_to_segment_map.clone(inp);
    fusion_segment->addInput(clone_tv);
    if (inp->isDefinitionType(ExprType::ViewOp)) {
      TORCH_INTERNAL_ASSERT(clone_tv != nullptr && clone_tv->isA<TensorView>());
      view_tvs.push_back(clone_tv->as<TensorView>());
    }
  }

  for (auto out : getAllOutputs(sg)) {
    fusion_segment->addOutput(complete_to_segment_map.clone(out));
  }

  for (auto tv : view_tvs) {
    tv->convertRfactorToRootDomain();
  }

  return fusion_segment;
}

void SegmentCandidateFinder::resetTraversal() {
  for (auto group : groups()) {
    // Start traversal at input groups
    if (group->producer_edges.empty()) {
      to_visit_.push_back(group);
    }
    group->visited_ = false;
    group->level_ = 0;
  }
}

void SegmentCandidateFinder::resetLevels() {
  while (!to_visit_.empty()) {
    auto visit = to_visit_.front();
    to_visit_.pop_front();

    // All inputs processed?
    bool ready = true;
    if (!visit->producer_edges.empty()) {
      ready = std::all_of(
          visit->producer_edges.begin(),
          visit->producer_edges.end(),
          [&](SegmentedEdge* dep) { return dep->from->visited_; });
    }

    if (!ready) {
      // In case traversal doesn't complete because there's an error in the
      // DAG topology.
      next_to_visit_.push_back(visit);
      continue;
    }

    visit->visited_ = true;

    to_visit_.insert(
        to_visit_.end(), next_to_visit_.begin(), next_to_visit_.end());
    next_to_visit_.clear();

    for (auto out : visit->consumer_edges) {
      to_visit_.push_back(out->to);
    }

    visit->level_ = 0;
    for (auto inp : visit->producer_edges) {
      visit->level_ = std::max(visit->level_, inp->from->level_ + 1);
    }
  }
  TORCH_INTERNAL_ASSERT(
      next_to_visit_.empty(), "Error in graph, is not a DAG.");
}

// Disconect group from neighbors, and return edges that were disconnected
std::unordered_set<SegmentedEdge*> SegmentCandidateFinder::disconnectGroup(
    SegmentedGroup* group) {
  std::unordered_set<SegmentedEdge*> removed_edges(
      group->producer_edges.begin(), group->producer_edges.end());

  for (auto edge : group->producer_edges) {
    auto from = edge->from;
    auto& from_edges = from->consumer_edges;
    auto from_edge_it = std::find(from_edges.begin(), from_edges.end(), edge);
    TORCH_INTERNAL_ASSERT(
        from_edge_it != from_edges.end(), "Could not find edge to remove.");
    from_edges.erase(from_edge_it);
  }

  for (auto edge : group->consumer_edges) {
    removed_edges.insert(edge);
    auto to = edge->to;
    auto& to_edges = to->producer_edges;
    auto to_edge_it = std::find(to_edges.begin(), to_edges.end(), edge);
    TORCH_INTERNAL_ASSERT(
        to_edge_it != to_edges.end(), "Could not find edge to remove.");
    to_edges.erase(to_edge_it);
  }

  group->producer_edges.clear();
  group->consumer_edges.clear();

  return removed_edges;
}

void SegmentCandidateFinder::eraseGroups(
    std::unordered_set<SegmentedGroup*>& groups_to_erase) {
  std::unordered_set<SegmentedEdge*> edges_to_erase;
  for (auto group : groups_to_erase) {
    auto disconnected_edges = disconnectGroup(group);
    edges_to_erase.insert(disconnected_edges.begin(), disconnected_edges.end());
  }

  edges().erase(
      std::remove_if(
          edges().begin(),
          edges().end(),
          [&edges_to_erase](SegmentedEdge* edge) {
            if (edges_to_erase.find(edge) != edges_to_erase.end()) {
              return true;
            };
            return false;
          }),
      edges().end());

  groups().erase(
      std::remove_if(
          groups().begin(),
          groups().end(),
          [&groups_to_erase](SegmentedGroup* group) {
            if (groups_to_erase.find(group) != groups_to_erase.end()) {
              return true;
            };
            return false;
          }),
      groups().end());
}

SegmentedGroup* SegmentCandidateFinder::mergeNodes() {
  SegmentedGroup* last_merged = nullptr;
  auto it = to_merge_.begin();
  TORCH_INTERNAL_ASSERT(to_merge_.size() % 2 == 0);
  while (it != to_merge_.end()) {
    auto group1 = *it++;
    auto group2 = *it++;

    clean_up_groups_.emplace(group1);
    clean_up_groups_.emplace(group2);

    // Make the new joined node
    auto joined_group = segmented_fusion_->newGroup();

    joined_group->input_vals =
        uniqueValConcat({group1->input_vals, group2->input_vals});

    joined_group->output_vals =
        uniqueValConcat({group1->output_vals, group2->output_vals});

    joined_group->exprs_ = group1->exprs_;
    joined_group->exprs_.insert(
        joined_group->exprs_.end(),
        group2->exprs_.begin(),
        group2->exprs_.end());

    auto producer_edges = getMergedProducerEdges(group1, group2);
    // Connect joined group to resulting neighbors
    for (auto edge : producer_edges) {
      auto from = edge->from;
      auto val = edge->val;

      auto new_edge = segmented_fusion_->newEdge(from, joined_group, val);
      joined_group->producer_edges.push_back(new_edge);
      from->consumer_edges.push_back(new_edge);
    }

    auto consumer_edges = getMergedConsumerEdges(group1, group2);

    for (auto edge : consumer_edges) {
      auto to = edge->to;
      auto val = edge->val;

      auto new_edge = segmented_fusion_->newEdge(joined_group, to, val);
      joined_group->consumer_edges.push_back(new_edge);
      edge->to->producer_edges.push_back(new_edge);
    }

    joined_group->setHeuristic(deriveHeuristic(joined_group));
    // Need to maintain the group dependency data if it has been intialized
    //  by previous merging
    if (group_dependency_) {
      group_dependency_->as<GroupDependencyAnalysis>()->mergeGroups(
          group1, group2, joined_group);
    }
    last_merged = joined_group;
  }

  to_merge_.clear();
  for (auto group : clean_up_groups_) {
    auto disconnected_edges = disconnectGroup(group);
    clean_up_edges_.insert(
        disconnected_edges.begin(), disconnected_edges.end());
  }

  edges().erase(
      std::remove_if(
          edges().begin(),
          edges().end(),
          [this](SegmentedEdge* edge) {
            if (this->clean_up_edges_.find(edge) !=
                this->clean_up_edges_.end()) {
              return true;
            };
            return false;
          }),
      edges().end());

  groups().erase(
      std::remove_if(
          groups().begin(),
          groups().end(),
          [this](SegmentedGroup* group) {
            if (this->clean_up_groups_.find(group) !=
                this->clean_up_groups_.end()) {
              return true;
            };
            return false;
          }),
      groups().end());

  clean_up_edges_.clear();
  clean_up_groups_.clear();

  return last_merged;
}

// Logic largely parallels mergeNodes, but they are used
//  in different phases of segmentation. Should consider
//  a clean up and share the implementations.
SegmentedGroup* SegmentCandidateFinder::mergeAllGivenGroups(
    const std::vector<SegmentedGroup*>& groups_to_merge) {
  TORCH_INTERNAL_ASSERT(
      !groups_to_merge.empty(),
      "fusion segment :(mergeAllGivenGroups) tried to merge no groups")

  // Make a set to detect internal edges
  std::unordered_set<SegmentedGroup*> group_set(
      groups_to_merge.begin(), groups_to_merge.end());

  // Sets to de-duplicate multiple uses of
  //  input/edge values and re-computations of exprs
  std::unordered_set<Val*> used_edge_vals_set;
  std::unordered_set<Val*> used_input_vals_set;
  std::unordered_set<Expr*> exprs_set;

  // Create new group
  auto joined_group = segmented_fusion_->newGroup();

  // Populate edges, exprs, global vals
  //  from each of the groups
  for (auto group : groups_to_merge) {
    // Populate complete fusion inputs to the group
    for (auto input_val : group->input_vals) {
      if (!used_input_vals_set.count(input_val)) {
        used_input_vals_set.insert(input_val);
        joined_group->input_vals.push_back(input_val);
      }
    }

    // Populate complete fusion outputs from the group
    for (auto output_val : group->output_vals) {
      joined_group->output_vals.push_back(output_val);
    }

    // Populate producer edges to the group
    for (auto edge : group->producer_edges) {
      if (
          // Check this is not internal edge
          !group_set.count(edge->from) &&
          // Check this val has been added or not
          !used_edge_vals_set.count(edge->val)) {
        used_edge_vals_set.insert(edge->val);
        auto new_producer_edge =
            segmented_fusion_->newEdge(edge->from, joined_group, edge->val);
        joined_group->producer_edges.push_back(new_producer_edge);
        edge->from->consumer_edges.push_back(new_producer_edge);
      }
    }

    // Populate consumer edges from the group
    for (auto edge : group->consumer_edges) {
      if (
          // Check this is not internal edge
          !group_set.count(edge->to)) {
        auto new_consumer_edge =
            segmented_fusion_->newEdge(joined_group, edge->to, edge->val);
        joined_group->consumer_edges.push_back(new_consumer_edge);
        edge->to->producer_edges.push_back(new_consumer_edge);
      }
    }

    // Populate exprs
    for (auto expr : group->exprs_) {
      if (!exprs_set.count(expr)) {
        joined_group->exprs_.push_back(expr);
        exprs_set.insert(expr);
      }
    }
  }

  // Clean up original groups from segmented fusion
  for (auto group : groups_to_merge) {
    auto disconnected_edges = disconnectGroup(group);
    clean_up_edges_.insert(
        disconnected_edges.begin(), disconnected_edges.end());
  }

  edges().erase(
      std::remove_if(
          edges().begin(),
          edges().end(),
          [this](SegmentedEdge* edge) { return clean_up_edges_.count(edge); }),
      edges().end());

  groups().erase(
      std::remove_if(
          groups().begin(),
          groups().end(),
          [&group_set](SegmentedGroup* group) -> bool {
            return group_set.count(group);
          }),
      groups().end());

  clean_up_edges_.clear();

  joined_group->setHeuristic(deriveHeuristic(joined_group));
  return joined_group;
}
namespace {

// Guard to temporarily change the inputs and outputs of a fusion. On
// destruction will return fusion to original state.
// Not used temporarily but will be useful when adding more mergin heuristics
class FusionSegmentGuard : public NonCopyable {
 public:
  FusionSegmentGuard() = delete;

  FusionSegmentGuard(
      Fusion* fusion,
      std::vector<Val*> inputs,
      std::vector<Val*> outputs)
      : fusion_(fusion),
        old_inputs_(fusion->inputs()),
        old_outputs_(fusion->outputs()),
        new_inputs_(std::move(inputs)),
        new_outputs_(std::move(outputs)) {
    FUSER_PERF_SCOPE("Segmenter::FusionSegmentGuard");
    TORCH_INTERNAL_ASSERT(fusion_ != nullptr);
    for (auto old_inp : old_inputs_) {
      fusion_->removeInput(old_inp);
    }

    for (auto old_out : old_outputs_) {
      fusion_->removeOutput(old_out);
    }

    for (auto new_inp : new_inputs_) {
      fusion_->addInput(new_inp);
    }

    for (auto new_out : new_outputs_) {
      fusion_->addOutput(new_out);
    }
  }

  ~FusionSegmentGuard() {
    FUSER_PERF_SCOPE("~Segmenter::FusionSegmentGuard");

    if (fusion_ == nullptr) {
      return;
    }
    for (auto new_inp : new_inputs_) {
      fusion_->removeInput(new_inp);
    }

    for (auto new_out : new_outputs_) {
      fusion_->removeOutput(new_out);
    }

    for (auto old_inp : old_inputs_) {
      fusion_->addInput(old_inp);
    }

    for (auto old_out : old_outputs_) {
      fusion_->addOutput(old_out);
    }
  }

 private:
  Fusion* const fusion_ = nullptr;
  const std::vector<Val*> old_inputs_;
  const std::vector<Val*> old_outputs_;
  const std::vector<Val*> new_inputs_;
  const std::vector<Val*> new_outputs_;
};

c10::optional<ScheduleHeuristic> tryMerge(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    SegmentedGroup* a,
    SegmentedGroup* b = nullptr) {
  FusionSegmentGuard fsg(fusion, getAllInputs(a, b), getAllOutputs(a, b));

  scheduler_debug_utils::canScheduleMessage(
      "\n**Segmenter** Considering fusion:\n", fusion);
  return SchedulerEntry::proposeHeuristics(fusion, runtime_info);
}

c10::optional<ScheduleHeuristic> tryMerge(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<SegmentedGroup*>& segmented_groups) {
  FusionSegmentGuard fsg(
      fusion,
      allInputsIfTrueElseOutputs(segmented_groups, true),
      allInputsIfTrueElseOutputs(segmented_groups, false));
  scheduler_debug_utils::canScheduleMessage(
      "\n**Segmenter** Considering fusion:\n", fusion);
  return SchedulerEntry::proposeHeuristics(fusion, runtime_info);
}

// This function is for cleanup and
//  easier debugging. It shouldn't affect functionality
//  since segmented fusions are compiled with fusion
//  guard on the edges instead of actually looking
//  at the exprs.
void deDuplicateScalarExprs(std::vector<Expr*>& exprs) {
  // Exprs in SegmentedGroup are not ordered
  // so it is ok to insert them from unordered
  // set
  std::unordered_set<Expr*> scalar_expr_set;

  std::copy_if(
      exprs.begin(),
      exprs.end(),
      std::inserter(scalar_expr_set, scalar_expr_set.end()),
      [](Expr* expr) { return ir_utils::isScalarOp(expr); });

  if (!scalar_expr_set.empty()) {
    exprs.erase(
        std::remove_if(
            exprs.begin(),
            exprs.end(),
            [&scalar_expr_set](Expr* expr) {
              return scalar_expr_set.count(expr);
            }),
        exprs.end());
    exprs.insert(exprs.end(), scalar_expr_set.begin(), scalar_expr_set.end());
  }
}

} // namespace

c10::optional<std::unique_ptr<SchedulerEntry>> SegmentedGroup::
    getMaybeSchedulerEntry(SchedulerRuntimeInfo& runtime_info) {
  FUSER_PERF_SCOPE("SegmentedGroup::getMaybeSchedulerEntry");
  auto fusion = segmented_fusion_->completeFusion();
  auto data_cache = segmented_fusion_->getCachedHeuristicDataFor(this);
  FusionSegmentGuard fsg(fusion, getAllInputs(this), getAllOutputs(this));
  if (!SchedulerEntry::canSchedule(
          heuristic(), fusion, runtime_info, data_cache)) {
    return c10::nullopt;
  }
  return SchedulerEntry::makeEntry(
      heuristic(), fusion, runtime_info, data_cache);
}

void SegmentedGroup::resetExprList() {
  auto input_group_vec = getAllInputs(this);
  std::unordered_set<Val*> input_group_set(
      input_group_vec.begin(), input_group_vec.end());
  auto expr_set =
      DependencyCheck::getAllExprsBetween(input_group_set, getAllOutputs(this));
  exprs_ = std::vector<Expr*>(expr_set.begin(), expr_set.end());
}

// Custom merge node passes:
//  These passes are added at the beginning or the end of
//  the node merging process to direct the heuristics of
//  node merging process
//
//  Should consider generalization and make a proper interface
//   if we have more merge node heuristics like this

//! Translate Welford
//!
//! This pass can be inserted at any stages of segmentation,
//!  and it tries to replace welford ops with persistent
//!  mean and var ops.
//!
//! The checking of feasibility of persistent kernels
//!  is through normalization schedulers. The general idea
//!  is to first try to translate on a copy, and see if
//!  normalization scheduler is willing to produce a
//!  persistent kernel.
//!
//! For complete fusion this pass checks if all the
//!  welford ops can be translated simultaneously to
//!  produce a persistent normalization kernel and
//!  will perform translation if checks pass.
//!
//! For segmented fusion, same check is performed within
//!  each segmented group to collect applicable welford ops,
//!  and actual translations are performed on the complete
//!  fusion after all the checks are done.
class TranslateApplicableWelford {
 public:
  //! Try translation on each segmented group of
  //!  given segmented fusion
  //!  returns true if any welford has been translated
  static bool run(
      SegmentedFusion* segmented_fusion,
      const KernelArgumentHolder& runtime_inputs) {
    TranslateApplicableWelford translate_welford(
        segmented_fusion, runtime_inputs);
    return translate_welford.translated_any_welford_;
  }

  //! Try translation on complete fusion,
  //!  returns true if any welford has been translated
  static bool run(Fusion* fusion, const KernelArgumentHolder& runtime_inputs) {
    TranslateApplicableWelford translate_welford(fusion, runtime_inputs);
    return translate_welford.translated_any_welford_;
  }

 private:
  explicit TranslateApplicableWelford(
      SegmentedFusion* segmented_fusion,
      const KernelArgumentHolder& runtime_inputs);

  explicit TranslateApplicableWelford(
      Fusion* fusion,
      const KernelArgumentHolder& runtime_inputs);

  //! Given vector of welford ops from the same fusion,
  //!  checks if translating all of them result in a
  //!  persistent normalization kernel by try-runs on
  //!  a test copy of the original fusion.
  //!
  //! Supported use cases are either un-segmented fusion,
  //!  or all the given welfords are within the same
  //!  segmented group. In the latter case, the segmented
  //!  group containing all the welford ops needs to be
  //!  provided.
  bool wouldTranslateToPersistent(
      const std::vector<WelfordOp*>& orignal_welfords,
      SegmentedGroup* group = nullptr);

  //! Translate the given welford op into separate
  //! average and standard deviation calculation.
  void translateSingleWelford(WelfordOp* welford);

  //! Utility to test if a translated fusion
  //!  gives a persistent kernel. Uses normalization
  //!  scheduler to do the test.
  bool isValidPersistentFusion(
      Fusion* translated_fusion,
      SchedulerRuntimeInfo& runtime_info);

 private:
  //! Indicates any translation happened.
  bool translated_any_welford_ = false;

  //! a reference to global fusion runtime inputs
  const KernelArgumentHolder& runtime_inputs_;

  //! For translation within group only,
  //!  group boundary at test copy
  //! (see wouldTranslateToPersistent implementation )
  std::vector<Val*> test_group_inputs_;
  std::vector<Val*> test_group_outputs_;
};

TranslateApplicableWelford::TranslateApplicableWelford(
    Fusion* fusion,
    const KernelArgumentHolder& runtime_inputs)
    : runtime_inputs_(runtime_inputs) {
  auto exprs = fusion->exprs();
  std::vector<WelfordOp*> orignal_welfords(
      ir_utils::filterByType<WelfordOp>(exprs).begin(),
      ir_utils::filterByType<WelfordOp>(exprs).end());

  if (wouldTranslateToPersistent(orignal_welfords)) {
    for (auto welford : orignal_welfords) {
      translateSingleWelford(welford);
    }
    translated_any_welford_ = true;
  }
}

TranslateApplicableWelford::TranslateApplicableWelford(
    SegmentedFusion* segmented_fusion,
    const KernelArgumentHolder& runtime_inputs)
    : runtime_inputs_(runtime_inputs) {
  std::vector<SegmentedGroup*> translated_groups;
  std::vector<WelfordOp*> welford_to_translate;
  // Find welfords that can be translated in each group
  for (auto group : segmented_fusion->groups()) {
    std::vector<WelfordOp*> welford_in_group(
        ir_utils::filterByType<WelfordOp>(group->exprs()).begin(),
        ir_utils::filterByType<WelfordOp>(group->exprs()).end());

    if (wouldTranslateToPersistent(welford_in_group, group)) {
      translated_groups.push_back(group);
      welford_to_translate.insert(
          welford_to_translate.end(),
          welford_in_group.begin(),
          welford_in_group.end());
    }
  }

  // Actually translate the welford ops
  // and record all the vals that have been
  // replaced by the translation.
  for (auto welford : welford_to_translate) {
    translateSingleWelford(welford);
  }

  for (auto translated_group : translated_groups) {
    // Update heuristics and expr list of translated groups
    translated_group->heuristic_ = ScheduleHeuristic::Persistent;
    translated_group->resetExprList();
  }
}

bool TranslateApplicableWelford::isValidPersistentFusion(
    Fusion* translated_fusion,
    SchedulerRuntimeInfo& runtime_info) {
  if (!SchedulerEntry::canSchedule(
          ScheduleHeuristic::Persistent, translated_fusion, runtime_info)) {
    return false;
  }

  auto scheduler = SchedulerEntry::makeEntry(
      ScheduleHeuristic::Persistent, translated_fusion, runtime_info);

  return scheduler->reductionParams().persistent_kernel;
}

bool TranslateApplicableWelford::wouldTranslateToPersistent(
    const std::vector<WelfordOp*>& orignal_welfords,
    SegmentedGroup* group) {
  if (orignal_welfords.empty()) {
    return false;
  }

  // Make sure all welford ops come from the same complete fusion
  auto fusion = orignal_welfords[0]->fusion();
  TORCH_INTERNAL_ASSERT(
      std::all_of(
          orignal_welfords.begin(),
          orignal_welfords.end(),
          [fusion](WelfordOp* welford) { return welford->fusion() == fusion; }),
      "Welfords in given vector not in the same fusion");

  // Make initial `in-progress copy`
  auto test_copy = std::make_unique<Fusion>();
  auto original_to_test_map = Fusion::copy(fusion, test_copy.get());

  std::vector<WelfordOp*> copied_welfords;
  std::transform(
      orignal_welfords.begin(),
      orignal_welfords.end(),
      std::back_inserter(copied_welfords),
      [&original_to_test_map](auto welford) {
        return original_to_test_map.clone(welford);
      });
  // Copied welfords will be invalidated on translation, but Vals will be
  // reused, keep a reference to them.
  std::vector<Val*> welford_avgs;
  std::vector<Val*> welford_vars;
  for (auto welford : copied_welfords) {
    welford_avgs.push_back(welford->outAvg());
    welford_vars.push_back(welford->outVar());
  }

  // Translate the welford ops
  for (auto welford_to_translate : copied_welfords) {
    translateSingleWelford(welford_to_translate);
  }

  SchedulerRuntimeInfo runtime_info(test_copy.get(), runtime_inputs_, true);
  // If we are looking at a segment of fusion,
  //  we maintain the segmented group boundary,
  //  one set for in_progress copy and one set
  //  for `test copy`
  if (group != nullptr) {
    auto original_inputs = getAllInputs(group);
    auto original_outputs = getAllOutputs(group);
    test_group_inputs_.clear();
    test_group_outputs_.clear();
    std::transform(
        original_inputs.begin(),
        original_inputs.end(),
        std::back_inserter(test_group_inputs_),
        [&original_to_test_map](Val* in) {
          return original_to_test_map.clone(in);
        });
    std::transform(
        original_outputs.begin(),
        original_outputs.end(),
        std::back_inserter(test_group_outputs_),
        [&original_to_test_map](Val* out) {
          return original_to_test_map.clone(out);
        });

    // If only average is used from welford, we should still translate, but we
    // might not detect persistence if variance isn't actually used/marked as an
    // output in the test.
    for (auto outs_i : c10::irange(welford_avgs.size())) {
      auto avg = welford_avgs[outs_i];
      auto var = welford_vars[outs_i];
      if (avg->uses().empty()) {
        test_group_outputs_.push_back(avg);
      }

      if (var->uses().empty()) {
        test_group_outputs_.push_back(var);
      }
    }

    // Temporarily localize test copy around
    //  the group boundary
    FusionSegmentGuard fsg(
        test_copy.get(), test_group_inputs_, test_group_outputs_);

    // Test if the translated copy is persistent
    return isValidPersistentFusion(test_copy.get(), runtime_info);
  }
  // In the case where we work on un-segmented
  //  fusion, no group boundary logic, just
  //  translate and test.
  return isValidPersistentFusion(test_copy.get(), runtime_info);
}

void TranslateApplicableWelford::translateSingleWelford(WelfordOp* welford) {
  auto fusion = welford->fusion();
  FusionGuard fg(fusion);
  // Only support translation of welford ops that
  // doesn't take inputs that are already statistics,
  // i.e. an r-factor product.
  // This translation works on un-scheduled fusions so
  //  shouldn't expect to see this.
  TORCH_INTERNAL_ASSERT(welford->inN()->isOneInt());

  // Grab the inputs and outputs of the welford
  auto in_val = welford->in()->as<TensorView>();
  auto out_avg = welford->outAvg()->as<TensorView>();
  auto out_var = welford->outVar()->as<TensorView>();
  auto out_N = welford->outN()->as<TensorView>();

  fusion->removeExpr(welford);
  // Not safe to use welford anymore
  welford = nullptr;

  // Create normalization based welford graph
  //  largely taken from batchnorm cpp benchmark
  const auto& in_root =
      TensorDomain::noReductions(in_val->getMaybeRFactorDomain());
  const auto& out_root = out_avg->getRootDomain();
  std::vector<int> red_axes;

  TORCH_INTERNAL_ASSERT(
      in_root.size() == out_root.size(),
      "Invalid root domains of Welford input and output.",
      " Input: ",
      ir_utils::toString(in_root),
      ". Output: ",
      ir_utils::toString(out_root));

  // Create scalar version of the feature element
  //  counting.
  Val* num_features = IrBuilder::create<Double>(1);
  std::vector<bool> broadcast_mask(in_root.size(), false);
  for (const auto i : c10::irange(in_root.size())) {
    if (out_root.at(i)->isReduction()) {
      red_axes.push_back(i);
      broadcast_mask[i] = true;
      num_features = mul(num_features, out_root.at(i)->extent());
    }
  }

  // Build a normalization expression group that is
  //  equivalent to a welford operation.
  auto x_sum = sum(in_val, red_axes);
  IrBuilder::create<BinaryOp>(BinaryOpType::Div, out_avg, x_sum, num_features);
  // welford.avg may be broadcast. Reuse it if found.
  TensorView* x_avg_bcast = nullptr;
  for (auto& use_expr : out_avg->uses()) {
    if (auto bcast = dynamic_cast<BroadcastOp*>(use_expr)) {
      if (bcast->getBroadcastDimFlags() == broadcast_mask) {
        // Same broadcast found.
        x_avg_bcast = bcast->out()->as<TensorView>();
        break;
      }
    }
  }

  // x_mean_sub may already exist. Reuse it if found.
  TensorView* x_mean_sub = nullptr;
  if (x_avg_bcast != nullptr) {
    for (auto& use_expr : x_avg_bcast->uses()) {
      if (auto bop = dynamic_cast<BinaryOp*>(use_expr)) {
        if (bop->getBinaryOpType() == BinaryOpType::Sub) {
          if (bop->lhs() == in_val && bop->rhs() == x_avg_bcast) {
            x_mean_sub = bop->out()->as<TensorView>();
          }
        }
      }
    }
  }

  if (x_avg_bcast == nullptr) {
    x_avg_bcast = broadcast(out_avg, broadcast_mask);
  }

  if (x_mean_sub == nullptr) {
    x_mean_sub = sub(in_val, x_avg_bcast);
  }

  auto x_mean_sub_pow = mul(x_mean_sub, x_mean_sub);
  IrBuilder::create<ReductionOp>(
      BinaryOpType::Add,
      IrBuilder::create<Double>(0.0),
      out_var,
      x_mean_sub_pow);
  IrBuilder::create<UnaryOp>(UnaryOpType::Set, out_N, num_features);

  // out_avg, out_N are now outputs of a pointwise ops and we
  //  need to clear out its reduction domains.
  out_avg->clearReductionIterDomains();
  out_N->clearReductionIterDomains();
}

bool SegmentCandidateFinder::TranslateWelfordInFusion(
    Fusion* fusion,
    const KernelArgumentHolder& runtime_inputs) {
  return TranslateApplicableWelford::run(fusion, runtime_inputs);
}

//! CombineReductions:
//!  This pass works before the main merge node process
//!    It identifies reduction operations that can be combined
//!    together to form a normalization kernel.
//!  Two reductions are considered the same type if they have
//!   the same root domain length, and the reduction axis are the same.
//!   This pass tries to merge nodes with the same reduction type based
//!   on the graph structure.
class CombineReductions {
  using GroupVec = std::vector<SegmentedGroup*>;
  class ReductionSignature;

 public:
  static void run(SegmentCandidateFinder* segment_candidate_finder) {
    CombineReductions combine_reductions(segment_candidate_finder);
  }
  static bool shouldRun(SegmentCandidateFinder* segment_candidate_finder);

 private:
  CombineReductions(SegmentCandidateFinder* segment_candidate_finder)
      : segment_candidate_finder_(segment_candidate_finder) {
    // Run pass over the segments

    // Collect segmented groups with reductions in them,
    //  Assuming running before any merge happened, so
    //  should see exactly one non-trivial reduction in each group
    for (auto group : segment_candidate_finder_->groups()) {
      if (auto rop_signature =
              ReductionSignature::makeReductionSignature(group)) {
        // Ignore pure squeeze operations in this analysis
        if (!rop_signature->hasNonTrivialReduction()) {
          continue;
        }

        groups_with_reductions_.push_back(group);
        // Check if this reduction signature is one that we have seen before
        auto signature_match_it = std::find_if(
            known_reduction_signatures_.begin(),
            known_reduction_signatures_.end(),
            [&rop_signature](auto& know_signature) {
              return know_signature->sameAs(rop_signature.get());
            });
        // Unmatched: Create a new signature entry if not known
        if (signature_match_it == known_reduction_signatures_.end()) {
          group_reduction_signature_map_[group] = rop_signature.get();
          known_reduction_signatures_.emplace_back(std::move(rop_signature));
        } else {
          // Matched known signature: Mark that this groups belongs to know
          // signature
          group_reduction_signature_map_[group] = signature_match_it->get();
        }
      }
    }

    // Keep trying to merge groups with compatible reductions and compatible
    // paths
    //  until no more merge opportunity can be identified
    bool merged_groups = true;
    while (merged_groups) {
      merged_groups = false;

      // Merge one pair of reduction groups at a time, and need
      //  the pass to update dependency info along the way to avoid cycles
      for (const auto first_group_index :
           c10::irange(groups_with_reductions_.size())) {
        if (merged_groups) {
          // Need to break and re-enter this loop because
          // groups_with_reductions_ will be updated
          break;
        }

        // Select one of the group to merge and get its reduction signature
        auto first_group = groups_with_reductions_[first_group_index];
        auto first_group_signature =
            group_reduction_signature_map_.at(first_group);

        for (const auto second_group_index : c10::irange(
                 first_group_index + 1, groups_with_reductions_.size())) {
          if (merged_groups) {
            // Need to break and re-enter this loop because
            // groups_with_reductions_ will be updated
            break;
          }
          auto second_group = groups_with_reductions_[second_group_index];
          auto second_group_signature =
              group_reduction_signature_map_.at(second_group);

          // Cannot merge if their signatures are not the same
          if (!first_group_signature->sameAs(second_group_signature)) {
            continue;
          }

          // first try a vertical merge
          merged_groups =
              verticalReductionMerge(first_group, second_group) != nullptr;
          if (!merged_groups) {
            // vertical merge didn't happen, try a horizontal merge
            merged_groups =
                horizontalReductionMerge(first_group, second_group) != nullptr;
          }
        }
      }
    }
  }

  //! Merge a vertical pair of producers and consumers,
  //!  the resulting group will include all nodes that are
  //!  also consumers of producer and producers of consumer,
  //!  i.e. values between the given producer-consumer pair.
  //!  Can be proven that:
  //!   1. Including all of these nodes will be cycle-free
  //!   2. These nodes are the minimal set of nodes to include if
  //!  for producer-consumer pair to be in the same group cycle-free
  //!
  //!  Returns nullptr if such merge cannot be achieved.
  //!  Reasons for not merging will include:
  //!   1. Given groups do not form producer-consumer pair
  //!   2. Merge will create cycle on the graph
  //!   3. The merged joined group cannot be scheduled
  SegmentedGroup* verticalReductionMerge(
      SegmentedGroup* first_group,
      SegmentedGroup* second_group) {
    // This is part of ReductionCombine pass, and we should only call this
    // function on a pair of reduction/normalization groups
    TORCH_INTERNAL_ASSERT(
        group_reduction_signature_map_.at(first_group)
            ->sameAs(group_reduction_signature_map_.at(second_group)));
    TORCH_INTERNAL_ASSERT(first_group != second_group);
    // Get the group dependency data from segment finder
    auto dependency_analysis = segment_candidate_finder_->getGroupDependency();

    // Check producer-consumer relationship
    SegmentedGroup* producer = nullptr;
    SegmentedGroup* consumer = nullptr;
    if (dependency_analysis->isConsumerOf(first_group, second_group)) {
      producer = second_group;
      consumer = first_group;
    } else if (dependency_analysis->isProducerOf(first_group, second_group)) {
      producer = first_group;
      consumer = second_group;
    } else {
      // Given groups aren't producer-consumer pair, won't merge
      return nullptr;
    }

    // Collect all groups that we need to merge along with the producer and
    // consumer
    auto all_groups_to_merge =
        getValidMinVerticalMergedGroupSet(producer, consumer);

    if (all_groups_to_merge.empty()) {
      // The vertical paths from producer to consumer have in-compatible
      // reductions
      //   so this vertical merge cannot be done.
      return nullptr;
    }

    // TODO: this step would not be deterministic, because valuesBetween isn't
    //       could fix this by a topological order
    std::vector<SegmentedGroup*> all_groups_to_merge_vec(
        all_groups_to_merge.begin(), all_groups_to_merge.end());

    // Final sanity check: the merged group can actually be scheduled
    Fusion* fusion =
        segment_candidate_finder_->segmented_fusion_->completeFusion();
    if (!tryMerge(
            fusion,
            segment_candidate_finder_->runtimeInfo(),
            all_groups_to_merge_vec)) {
      return nullptr;
    }

    // Merge this group
    auto joined_group =
        segment_candidate_finder_->mergeAllGivenGroups(all_groups_to_merge_vec);

    // Update dependency analysis
    dependency_analysis->mergeGroups(all_groups_to_merge, joined_group);

    // Update the reduction groups that are merged
    groups_with_reductions_.push_back(joined_group);
    group_reduction_signature_map_[joined_group] =
        group_reduction_signature_map_.at(first_group);
    groups_with_reductions_.erase(
        std::remove_if(
            groups_with_reductions_.begin(),
            groups_with_reductions_.end(),
            [&all_groups_to_merge](SegmentedGroup* group) {
              return all_groups_to_merge.has(group);
            }),
        groups_with_reductions_.end());

    return joined_group;
  }

  //! Horizontal reduction merging:
  //!  merge two horizontal groups with reduction expressions to make a joined
  //!  normalization group. A pair of horizontal groups are ones that are not
  //!  a producer-consumer pair, and share either a common producer or a common
  //!  consumer.
  //!
  //!  TODO: This implementation looks at common producers only, since common
  //!  consumers are not computed easily with current dependency analysis.
  SegmentedGroup* horizontalReductionMerge(
      SegmentedGroup* first_group,
      SegmentedGroup* second_group) {
    // This is part of ReductionCombine pass, and we should only call this
    // function on a pair of
    //  reduction/normalization groups
    TORCH_INTERNAL_ASSERT(
        group_reduction_signature_map_.at(first_group)
            ->sameAs(group_reduction_signature_map_.at(second_group)));
    TORCH_INTERNAL_ASSERT(first_group != second_group);

    auto dependency_analysis = segment_candidate_finder_->getGroupDependency();

    // Check that the two groups are not producer-consumer's
    if (dependency_analysis->isConsumerOf(first_group, second_group) ||
        dependency_analysis->isProducerOf(first_group, second_group)) {
      // This merge pass will not handle producer-consumer pairs
      return nullptr;
    }

    // Get common producers of the two group
    auto common_producers_set =
        dependency_analysis->getCommonProducersOf({first_group, second_group});
    if (common_producers_set.empty()) {
      // The given pair doesn't have a common producer.
      //  Either they have a common consumer, which we don't handle for now,
      //  or maybe the two given groups are not connected.
      return nullptr;
    }

    // We are looking for a very specific patterns here. The cases that this
    //  pattern will not capture are ones that reductions of different
    //  signatures are so interleaved that we cannot find a clear cut as
    //  explained below, without graph rewriting. Some graph re-writing on the
    //  segmented groups level could provide extra merging opportunities for
    //  free, which could be part of next step.
    //
    // The specific pattern we look for contains a common producer P with
    // immediate consumers C1, C2 such that all paths from C1 to first_group and
    // all paths from C2 to second_group won't hit a reduction with a different
    // signature.

    // Topologically sort the common producers and start with the topologically
    // minimal,
    //  i.e. one that are closest to the two groups. This will cut the search
    //  space.
    std::vector<SegmentedGroup*> common_producers;
    for (auto producer : common_producers_set) {
      if (!std::any_of(
              common_producers_set.begin(),
              common_producers_set.end(),
              [dependency_analysis, producer](SegmentedGroup* group) {
                return dependency_analysis->isProducerOf(producer, group);
              })) {
        common_producers.push_back(producer);
      }
    }

    // Visit the common producers found, starting from topologically minimum,
    // i.e. the ones closer to the groups
    for (auto common_producer : common_producers) {
      // Visit this common producer
      // Use a double loop in case the schedulers like some patterns
      //  better than the other
      for (auto first_consumer_edge : common_producer->consumer_edges) {
        auto producer_of_first_group = first_consumer_edge->to;
        auto to_merge_with_first_group = getValidMinVerticalMergedGroupSet(
            producer_of_first_group, first_group);
        if (to_merge_with_first_group.empty()) {
          // There's no valid merge path from this consumer of common producer,
          //  either due to a conflicting reduction signature, or simply there's
          //  no path to first group
          continue;
        }
        TORCH_INTERNAL_ASSERT(!dependency_analysis->isProducerOf(
            producer_of_first_group, second_group));
        for (auto second_consumer_edge : common_producer->consumer_edges) {
          auto producer_of_second_group = second_consumer_edge->to;
          auto to_merge_with_second_group = getValidMinVerticalMergedGroupSet(
              producer_of_second_group, second_group);
          if (to_merge_with_second_group.empty()) {
            // There's no valid merge path from this consumer of common
            // producer,
            //  either due to a conflicting reduction signature, or simply
            //  there's no path to second group
            continue;
          }
          TORCH_INTERNAL_ASSERT(!dependency_analysis->isProducerOf(
              producer_of_second_group, first_group));
          // At this point we should have a pair of valid candidates,final check
          // is to see if the combined group
          //  can be scheduled by schedulers
          // merge the two paths and de-duplicate,
          //  re-using container here with to_merge_with_second_group
          auto& groups_to_merge_set = to_merge_with_second_group;
          groups_to_merge_set.insert(
              to_merge_with_first_group.begin(),
              to_merge_with_first_group.end());
          std::vector<SegmentedGroup*> groups_to_merge_vec(
              groups_to_merge_set.begin(), groups_to_merge_set.end());
          Fusion* fusion =
              segment_candidate_finder_->segmented_fusion_->completeFusion();
          if (tryMerge(
                  fusion,
                  segment_candidate_finder_->runtimeInfo(),
                  groups_to_merge_vec)) {
            // Found a valid horizontal merge, want to proceed with merging here
            auto joined_group = segment_candidate_finder_->mergeAllGivenGroups(
                groups_to_merge_vec);
            dependency_analysis->mergeGroups(groups_to_merge_set, joined_group);

            groups_with_reductions_.push_back(joined_group);
            group_reduction_signature_map_[joined_group] =
                group_reduction_signature_map_.at(first_group);
            groups_with_reductions_.erase(
                std::remove_if(
                    groups_with_reductions_.begin(),
                    groups_with_reductions_.end(),
                    [&groups_to_merge_set](SegmentedGroup* group) {
                      return groups_to_merge_set.has(group);
                    }),
                groups_with_reductions_.end());

            return joined_group;
          }
        }
      }
    }

    // Searched all possibilities and there is no valid horizontal merge pattern
    //  found.
    return nullptr;
  }

  //! This is a utility method that is used in both vertical merging and
  //! horizontal merging.
  //!  It is used to identify the smallest set of groups to merge vertically
  //!  involving the
  //!   two given nodes.
  //!  Given a pair of nodes this utility distinguishes 3 cases:
  //!   1. if maybe_producer is the same as maybe_consumer, then returns
  //!   {maybe_producer}
  //!   2. if maybe_producer is actually a producer of consumer, returns a set
  //!   containing
  //!     the smallest merged group that would contain producer and consumer and
  //!     would not introduce a cycle. Returns empty set if such group has
  //!     a conflicting reduction signature.
  //!   3. returns empty set if neither conditions above apply.
  GroupSet getValidMinVerticalMergedGroupSet(
      SegmentedGroup* maybe_producer,
      SegmentedGroup* maybe_consumer) {
    auto dependency_analysis = segment_candidate_finder_->getGroupDependency();
    if (maybe_consumer == maybe_producer) {
      // maybe producer is the same as maybe_consumer
      return {maybe_consumer};
    } else if (dependency_analysis->isConsumerOf(
                   maybe_consumer, maybe_producer)) {
      auto groups_to_check =
          dependency_analysis->valuesBetween(maybe_producer, maybe_consumer);
      groups_to_check.pushBack(maybe_producer);
      groups_to_check.pushBack(maybe_consumer);

      // Check that either no group has a reduction or all groups have the same
      // reduction signature
      ReductionSignature* reduction_signature = nullptr;

      // Iterate through the minimal group set to see if any conflicts
      for (auto group : groups_to_check) {
        // Check that this group does not involve a output edge contraction
        //  This pass is intended to be a pre-merging pass. Since contracting an
        //   output edge does not generate much saving of global memory access
        //   we want to postpone merging these edges till the very final pass
        for (auto producer_edge_of_group : group->producer_edges) {
          if (groups_to_check.has(producer_edge_of_group->from) &&
              producer_edge_of_group->val->isFusionOutput()) {
            return {};
          }
        }
        for (auto consumer_edge_of_group : group->consumer_edges) {
          if (groups_to_check.has(consumer_edge_of_group->to) &&
              consumer_edge_of_group->val->isFusionOutput()) {
            return {};
          }
        }

        // Check that this group does not have a conflicting reduction signature
        if (group_reduction_signature_map_.count(group)) {
          if (reduction_signature != nullptr) {
            if (!group_reduction_signature_map_.at(group)->sameAs(
                    reduction_signature)) {
              // Found a conflict in reduction signature, cannot do a vertical
              // merge
              return {};
            }
          } else {
            reduction_signature = group_reduction_signature_map_.at(group);
          }
        }
      }
      return groups_to_check;
    }
    // maybe producer is not a producer of maybe consumer
    return {};
  }

 private:
  SegmentCandidateFinder* segment_candidate_finder_;

  // Wrapper class for reduction type
  //  Assuming there wouldn't be too many of them
  //  so won't need to create a hash
  // TODO:
  //   Want to reconsider this for transpose operations,
  //   need refactoring to handle reduction fusions across a transpose operation
  class ReductionSignature {
   public:
    bool sameAs(const ReductionSignature* reduction_signature) {
      if (reduction_signature == this) {
        return true;
      }

      if (root_domain_size_ != reduction_signature->root_domain_size_ ||
          has_nontrivial_reduction_ !=
              reduction_signature->has_nontrivial_reduction_ ||
          reduction_axes_.size() !=
              reduction_signature->reduction_axes_.size()) {
        return false;
      }

      for (const auto i : c10::irange(reduction_axes_.size())) {
        if (reduction_axes_[i] != reduction_signature->reduction_axes_[i]) {
          return false;
        }
      }

      return true;
    }

    bool sameAs(const ReductionSignature& reduction_signature) {
      return sameAs(&reduction_signature);
    }

    bool hasNonTrivialReduction() const {
      return has_nontrivial_reduction_;
    }

    static std::unique_ptr<ReductionSignature> makeReductionSignature(
        SegmentedGroup* group) {
      std::unique_ptr<ReductionSignature> signature = nullptr;

      for (auto expr : group->exprs()) {
        std::unique_ptr<ReductionSignature> new_signature = nullptr;

        if (auto rop = dynamic_cast<ReductionOp*>(expr)) {
          new_signature = std::make_unique<ReductionSignature>(rop);
        }
        if (auto wop = dynamic_cast<WelfordOp*>(expr)) {
          new_signature = std::make_unique<ReductionSignature>(wop);
        }

        if (new_signature != nullptr) {
          TORCH_INTERNAL_ASSERT(
              signature == nullptr || !signature->has_nontrivial_reduction_ ||
                  !new_signature->has_nontrivial_reduction_ ||
                  signature->sameAs(new_signature.get()),
              "Conflicting signature found in this group");
          signature = std::move(new_signature);
        }
      }
      return signature;
    }

    template <typename REDUCTION = ReductionOp>
    ReductionSignature(REDUCTION* rop) {
      auto out_tv = rop->out()->template as<TensorView>();
      has_nontrivial_reduction_ = out_tv->hasReduction();
      TORCH_INTERNAL_ASSERT(out_tv != nullptr);
      auto& root_domain = out_tv->getRootDomain();
      root_domain_size_ = root_domain.size();

      // Trivial reduction i.e. squeeze is tricky here:
      //  this pass doesn't want to touch any pure squeeze, i.e.:
      //    T0 [R(1), I(i0), I(i1)]
      //  meanwhile, for two reductions having
      //  squeezes, we do require they have squeeze at the
      //  same position so that they can be easily root domain mapped
      //  So T0 and T1 are the same signature,
      //    T0 [R(1), R(i0), I(i1)]
      //    T1 [R(1), R(i0), I(i1)]
      //  but T2 and T3 below are not
      //    T0 [R(1), R(1), R(i0), I(i1)]
      //    T1 [R(1), R(i0), I(i1)]
      for (const auto i : c10::irange(root_domain_size_)) {
        if (root_domain[i]->isReduction()) {
          reduction_axes_.push_back(i);
        }
        if (!root_domain[i]->isTrivialReduction()) {
          has_nontrivial_reduction_ = true;
        }
      }
    }

   private:
    size_t root_domain_size_ = 0;
    std::vector<int> reduction_axes_;
    bool has_nontrivial_reduction_ = false;
  };

  //! Keeps track of groups with reduction expressions,
  //!  using a vector here to maintain a deterministic ordering
  GroupVec groups_with_reductions_;

  //! Maps groups to their corresponding signature type
  std::unordered_map<SegmentedGroup*, ReductionSignature*>
      group_reduction_signature_map_;

  //! Maintains all reduction signatures seen in the segmented fusion
  std::vector<std::unique_ptr<ReductionSignature>> known_reduction_signatures_;
};

//! This is to be checked
bool CombineReductions::shouldRun(
    SegmentCandidateFinder* segment_candidate_finder) {
  std::vector<std::unique_ptr<ReductionSignature>> known_reductions;
  // Iterate over group segments we have before segment candidate finder
  //  tries to merge any groups
  for (auto group : segment_candidate_finder->groups()) {
    if (auto reduction_signature =
            ReductionSignature::makeReductionSignature(group)) {
      if (reduction_signature->hasNonTrivialReduction() &&
          std::any_of(
              known_reductions.begin(),
              known_reductions.end(),
              [&reduction_signature](auto& know_signature) {
                return know_signature->sameAs(reduction_signature.get());
              })) {
        // Found two reductions with the same signature, run pass
        return true;
      }
      known_reductions.emplace_back(std::move(reduction_signature));
    }
  }
  return false;
}

namespace {

//! Returns true if group1 and group2 are an immediate producer-consumer pair.
bool areDirectlyConnected(SegmentedGroup* group1, SegmentedGroup* group2) {
  // Check if group1 is a immediate consumer of group2
  if (std::any_of(
          group1->producer_edges.begin(),
          group1->producer_edges.end(),
          [group2](SegmentedEdge* edge) { return edge->from == group2; })) {
    return true;
  }

  // Check if group1 is a immediate producer of group2
  if (std::any_of(
          group1->consumer_edges.begin(),
          group1->consumer_edges.end(),
          [group2](SegmentedEdge* edge) { return edge->to == group2; })) {
    return true;
  }

  return false;
}

} // namespace

bool SegmentCandidateFinder::codeGenSupportedMerge(
    SegmentedGroup* group1,
    SegmentedGroup* group2) {
  TORCH_INTERNAL_ASSERT(
      areDirectlyConnected(group1, group2),
      "only support testing immediate producer-consumer groups");
  Fusion* fusion = segmented_fusion_->completeFusion();
  auto h = tryMerge(fusion, runtime_info_, group1, group2);
  return h.has_value();
}

// TODO: consider caching the heuristics value so tryMerge doesn't have to be
//       called twice
ScheduleHeuristic SegmentCandidateFinder::deriveHeuristic(
    SegmentedGroup* group) {
  Fusion* fusion = segmented_fusion_->completeFusion();
  auto h = tryMerge(fusion, runtime_info_, group);
  TORCH_INTERNAL_ASSERT(h.has_value());
  return h.value();
}

SegmentCandidateFinder::SegmentCandidateFinder(
    std::unique_ptr<Fusion> fusion,
    const KernelArgumentHolder& inputs,
    SegmentCandidateFinderOptions options)
    : options_(options),
      runtime_info_(fusion.get(), inputs, true),
      runtime_inputs_(inputs) {
  segmented_fusion_ = std::make_unique<SegmentedFusion>(std::move(fusion));
  findSegments();
}

void SegmentCandidateFinder::findSegments() {
  FUSER_PERF_SCOPE("Finding valid fusion segment solutions");
  // TODO: Make traversal items local to this function.

  // Need this for initialization of the DAG that is process
  std::unordered_map<Expr*, SegmentedGroup*> expr2group;

  // Keep track of complete fusion input use
  std::unordered_map<Val*, SegmentedGroup*> input2group;

  // Initialize DAG, convert each expr to a segment group
  auto exprs = completeFusion()->exprs();
  for (auto expr : exprs) {
    if (!ir_utils::isScalarOp(expr)) {
      auto new_group = segmented_fusion_->newGroup(expr);
      expr2group.insert(std::make_pair(expr, new_group));
    }
  }

  // Find all expresions that are simply unary ops from inputs. Don't segment
  // these as they're easy targets for recomputation. Only go until the first
  // expression that has multiple uses. We could continue, but the logic of
  // hacking the fusion "inputs" logic gets a bit more complicated.

  // Expressions to exclude from segmentation because they're just derived from
  // unary ops on inputs to the complete fusion
  VectorOfUniqueEntries<Expr*> excluded_inp_unary_exprs;

  // "Terminating" outputs from the excluded input unary exprs, these will be
  // treated as complete fusion inputs.
  VectorOfUniqueEntries<Val*> forwarded_inputs;
  {
    std::deque<Expr*> to_visit;
    for (auto inp : completeFusion()->inputs()) {
      if (std::all_of(inp->uses().begin(), inp->uses().end(), [](Expr* expr) {
            return expr->getExprType().value() == ExprType::UnaryOp;
          })) {
        to_visit.insert(to_visit.end(), inp->uses().begin(), inp->uses().end());
      }
    }

    while (!to_visit.empty()) {
      auto expr = to_visit.front();
      to_visit.pop_front();
      if (expr->getExprType().value() != ExprType::UnaryOp ||
          expr->output(0)->isFusionOutput()) {
        continue;
      }

      if (expr->output(0)->uses().size() > 1) {
        excluded_inp_unary_exprs.pushBack(expr);
        forwarded_inputs.pushBack(expr->output(0));
        continue;
      }

      to_visit.emplace_back(expr->output(0)->uses()[0]);
    }
  }

  auto excluded_fusion_inputs = IterVisitor::getInputsTo(
      {forwarded_inputs.begin(), forwarded_inputs.end()});

  // List of vals to treat as complete fusion inputs for segmentation
  auto forwarded_fusion_inputs = completeFusion()->inputs();

  forwarded_fusion_inputs.erase(
      std::remove_if(
          forwarded_fusion_inputs.begin(),
          forwarded_fusion_inputs.end(),
          [&excluded_fusion_inputs](Val* inp) {
            return std::find(
                       excluded_fusion_inputs.begin(),
                       excluded_fusion_inputs.end(),
                       inp) != excluded_fusion_inputs.end();
          }),
      forwarded_fusion_inputs.end());

  forwarded_fusion_inputs.insert(
      forwarded_fusion_inputs.end(),
      forwarded_inputs.begin(),
      forwarded_inputs.end());

  auto isFusionInput = [&forwarded_fusion_inputs](Val* val) -> bool {
    return std::find(
               forwarded_fusion_inputs.begin(),
               forwarded_fusion_inputs.end(),
               val) != forwarded_fusion_inputs.end();
  };

  // Insert auxiliary groups to use group dependency on inputs as well
  // TODO: these groups should never merged into any other groups, but are
  //       just there to support the dependency analysis. Later re-factor should
  //       avoid introducing them explicitly on the segmented fusion.
  for (auto input : forwarded_fusion_inputs) {
    // These groups are used to represent input as a common
    //  producer in horizontal merges, and should never be
    //  seen as a candidate for vertical merge
    auto new_group = segmented_fusion_->newGroup();
    input2group.insert({input, new_group});
  }

  // Create edges between the Exprs. Mark inputs and outputs of the fusion.
  for (auto expr : exprs) {
    // No group created for scalar ops
    if (ir_utils::isScalarOp(expr)) {
      continue;
    }

    if (excluded_inp_unary_exprs.has(expr)) {
      continue;
    }

    auto expr_group = expr2group.at(expr);
    for (auto inp : expr->inputs()) {
      if (isFusionInput(inp)) {
        expr_group->input_vals.push_back(inp);
        auto aux_group = input2group.at(inp);
        auto new_edge = segmented_fusion_->newEdge(aux_group, expr_group, inp);
        expr_group->producer_edges.push_back(new_edge);
        aux_group->consumer_edges.push_back(new_edge);
        continue;
      }

      // Could be something like a constant scalar, definition is nullptr, but
      // isn't an "input" to the fusion. At least not one provided by an
      // external source.
      if (inp->definition() == nullptr) {
        continue;
      }

      // No group created for scalar ops since they may need to be duplicated
      //  to avoid scalar edges. They are handled in resolveScalarsInGroup
      if (inp->isScalar()) {
        continue;
      }

      auto def_group = expr2group.at(inp->definition());
      auto new_edge = segmented_fusion_->newEdge(def_group, expr_group, inp);
      expr_group->producer_edges.push_back(new_edge);
      def_group->consumer_edges.push_back(new_edge);
    }
    for (auto out : expr->outputs()) {
      if (out->isFusionOutput()) {
        expr_group->output_vals.push_back(out);
      }
    }
  }

  auto reduction_ops = ir_utils::getReductionOps(
      segmented_fusion_->completeFusion(), true /* ignore_trivial */);
  auto welford_ops = ir_utils::filterByType<WelfordOp>(reduction_ops);

  if (options_.run_translate_welford &&
      (welford_ops.begin() != welford_ops.end())) {
    TranslateApplicableWelford::run(segmented_fusion_.get(), runtime_inputs_);
  }

  for (auto group : groups()) {
    if (!group->outputs().empty()) {
      // Set heuristics in case single reduction kernels were left out
      group->setHeuristic(deriveHeuristic(group));
    }
  }

  // Remove all scalar edges since they do not represent actual
  //  dependency among segmented groups.
  removeScalarEdges();

  // Run pre-merge heuristics
  if (options_.run_combine_reductions && CombineReductions::shouldRun(this)) {
    CombineReductions::run(this);
  }

  // All merges will be vertical beyond this point for now, so
  //  we can remove the input auxiliary groups. Should make the vertical
  //  merges avoid auxiliary group once we start general horizontal merges
  std::unordered_set<SegmentedGroup*> input_groups;
  for (auto input : forwarded_fusion_inputs) {
    input_groups.insert(input2group.at(input));
  }
  eraseGroups(input_groups);

  if (options_.run_herrmann_merge) {
    bool merged_nodes = true;
    // Initial merge iteration
    while (merged_nodes) {
      // Reset stateful traversal details in SegmentedGroups
      resetTraversal();

      resetLevels();

      for (auto& group : groups()) {
        if (group->merged_) {
          continue;
        }
        auto candidates = group->getMergeCandidates();
        if (candidates.empty()) {
          continue;
        }

        auto candidate_it = candidates.begin();
        while (candidate_it != candidates.end() &&
               !codeGenSupportedMerge(group, candidate_it->group)) {
          candidate_it++;
        }
        if (candidate_it == candidates.end()) {
          continue;
        }

        to_merge_.emplace_back(group);
        to_merge_.emplace_back(candidate_it->group);

        group->merged_ = true;
        group->merge_with_ = candidate_it->group;
        group->merge_through_ = candidate_it->edge;

        candidate_it->group->merged_ = true;
        candidate_it->group->merge_with_ = group;
        candidate_it->group->merge_through_ = candidate_it->edge;
      }

      if (to_merge_.empty()) {
        merged_nodes = false;
      }

      mergeNodes();
    }
  }

  if (options_.run_final_merge) {
    // TODO: consider interleaving herrmman merge and bruteforce merge, as
    // bruteforce merge can introduce opportunities for more herrmann merge
    finalMerge();
  }

  finalize();

  if (isDebugDumpEnabled(DebugDumpOption::FusionSegmentsDrawing)) {
    segmented_fusion_->draw();
  }
}

void SegmentCandidateFinder::finalMerge() {
  auto producer_check = getGroupDependency();

  bool merged_nodes = true;
  while (merged_nodes) {
    // Iterate all groups and check if a group
    //  can merge with one of its consumers
    for (auto producer_group : groups()) {
      // Populate consumers and their corresponding consumer edges
      std::unordered_map<SegmentedGroup*, SegmentedEdge*> consumer_edge_map;
      std::vector<SegmentedGroup*> all_consumers_of_producer_group;
      for (auto consumer : producer_group->consumer_edges) {
        // Since this is the last fusion pass, we can enable fusion through
        // outputs. Priority of this was decreased because if the only
        // connection between groups is an output node, best case scenario we
        // can save a single pass in memory. Where if it wasn't an output it
        // would be two passes.
        consumer_edge_map.insert({consumer->to, consumer});
      }
      // Populate all consumers from the map to avoid duplicate
      std::transform(
          consumer_edge_map.begin(),
          consumer_edge_map.end(),
          std::back_inserter(all_consumers_of_producer_group),
          [](auto& it) { return it.first; });

      for (auto consumer : all_consumers_of_producer_group) {
        if (!producer_check->isConsumerOfAny(
                consumer, all_consumers_of_producer_group) &&
            codeGenSupportedMerge(producer_group, consumer)) {
          to_merge_.emplace_back(producer_group);
          to_merge_.emplace_back(consumer);
          producer_group->merged_ = true;
          producer_group->merge_with_ = consumer;
          producer_group->merge_through_ = consumer_edge_map.at(consumer);
          consumer->merged_ = true;
          consumer->merge_with_ = producer_group;
          consumer->merge_through_ = producer_group->merge_through_;
          break;
        }
      }

      // Only want to merge one pair at a time so break if found any
      if (!to_merge_.empty()) {
        break;
      }
    }

    if (to_merge_.empty()) {
      merged_nodes = false;
    } else {
      TORCH_INTERNAL_ASSERT(
          to_merge_.size() == 2, "merging more than 2 nodes in final iter");
      mergeNodes();
    }
  }
}

void SegmentCandidateFinder::resolveScalarsInGroup(SegmentedGroup* group) {
  std::vector<Val*> to_visit;
  std::unordered_set<Val*> visited;

  // Collect all scalar uses in the group
  for (auto expr : group->exprs()) {
    for (auto input : expr->inputs()) {
      if (input->isScalar()) {
        to_visit.push_back(input);
      }
    }
  }

  // Keep track of composite fusion inputs used in this group
  std::unordered_set<Val*> input_set(
      group->input_vals.begin(), group->input_vals.end());

  // Record and append all missing scalar exprs at the end.
  std::vector<Expr*> exprs_to_add;

  // Do a stack based traversal of the scalar ops to avoid
  //  combinatorial duplication of exprs.
  while (!to_visit.empty()) {
    auto stack_top_val = to_visit.back();
    if (visited.count(stack_top_val)) {
      to_visit.pop_back();
    } else if (stack_top_val->definition() == nullptr) {
      // A scalar without def can be a scalar, a tensor dim,
      //  or a composite fusion input
      // The first two cases are handled in finalize(),
      //  the last case needs to add new input_val to this group.
      visited.insert(stack_top_val);
      // If this is a composite fusion scalar input, make sure this group has it
      if (stack_top_val->isFusionInput() && !input_set.count(stack_top_val)) {
        group->input_vals.push_back(stack_top_val);
        input_set.insert(stack_top_val);
      }
      to_visit.pop_back();
    } else {
      // A scalar with an actual definition
      auto definition_expr = stack_top_val->definition();
      bool all_inputs_visited = true;
      // If any of the inputs are not visited, visit them first
      for (auto input : definition_expr->inputs()) {
        if (!visited.count(input)) {
          all_inputs_visited = false;
          to_visit.push_back(input);
        }
      }
      // This node is ready to be visited
      if (all_inputs_visited) {
        // Collect the defining expr to insert into group
        exprs_to_add.push_back(definition_expr);
        visited.insert(stack_top_val);
        to_visit.pop_back();
      }
    }
  }

  // Add all the defining expr to the group
  for (auto expr : exprs_to_add) {
    group->exprs_.push_back(expr);
  }
}

void SegmentCandidateFinder::resolveInputsInGroup(SegmentedGroup* group) {
  std::vector<Val*> to_visit;
  std::unordered_set<Val*> visited;

  // Collect all inputs to group that are not inputs of fusion
  for (auto input : group->inputs()) {
    if (!input->isFusionInput()) {
      to_visit.push_back(input);
    }
  }

  // Reset group inputs to real inputs
  group->input_vals = IterVisitor::getInputsTo(group->inputs());

  // Grab all expressions needed to produce to_visit
  auto input_exprs = StmtSort::getExprs(completeFusion(), to_visit);

  // Insert those expressions at the beginning of the group
  group->exprs_.insert(
      group->exprs_.begin(), input_exprs.begin(), input_exprs.end());
}

void SegmentCandidateFinder::removeScalarEdges() {
  // Remove all scalar edges between groups
  //  They may have been created by welford
  //   translation.
  //  we will not need them after scalar
  //  resolution
  auto remove_scalar_edges_from_vec = [](std::vector<SegmentedEdge*>& edges) {
    edges.erase(
        std::remove_if(
            edges.begin(),
            edges.end(),
            [](SegmentedEdge* segmented_edge) {
              return segmented_edge->val->isScalar();
            }),
        edges.end());
  };

  remove_scalar_edges_from_vec(edges());
  for (auto group : groups()) {
    remove_scalar_edges_from_vec(group->producer_edges);
    remove_scalar_edges_from_vec(group->consumer_edges);
  }
}

void SegmentCandidateFinder::finalize() {
  // Remove unconnected groups
  groups().erase(
      std::remove_if(
          groups().begin(),
          groups().end(),
          [](SegmentedGroup* sg) { return !sg->isConnected(); }),
      groups().end());

  // Add group labeling
  int i = 0;
  for (auto it = groups().begin(); it != groups().end(); it++, i++) {
    deDuplicateScalarExprs((*it)->exprs_);
    (*it)->setID(i);
  }

  // TODO: too many things are currently abstracted under the term
  //  finalize. Need to re-structure in a follow up.

  // Finalize connections between segmented groups
  segmented_fusion_->finalize();

  // Resolve all the scalar expressions needed in each group
  for (auto group : segmented_fusion_->groups()) {
    resolveScalarsInGroup(group);
  }

  // Resolve all the scalar expressions needed in each group
  for (auto group : segmented_fusion_->groups()) {
    resolveInputsInGroup(group);
  }

  // Finalize each group, fill in the missing inputs, i.e. tensor dims.
  for (auto g : groups()) {
    g->setHeuristic(deriveHeuristic(g));
    g->finalize();
  }
}

GroupDependencyAnalysis* SegmentCandidateFinder::getGroupDependency() {
  if (!group_dependency_) {
    group_dependency_ =
        std::make_unique<GroupDependencyAnalysis>(segmented_fusion_.get());
  }
  return group_dependency_->as<GroupDependencyAnalysis>();
}

FusionKernelRuntime::SchedulerEntryPtr SegmentedFusion::
    makeInitialSchedulerEntry(
        SegmentedGroup* sg,
        SchedulerRuntimeInfo& runtime_info) {
  auto local_fusion = completeFusion();
  FusionSegmentGuard fsg(local_fusion, getAllInputs(sg), getAllOutputs(sg));
  // This will be the first time each group is scheduled. So we'd want to
  //  construct the cache data here.
  auto data_cache_ptr = std::make_unique<HeuristicSummary>(
      local_fusion, sg->heuristic(), runtime_info);
  auto data_cache = data_cache_ptr.get();
  setCachedHeuristicDataFor(sg, std::move(data_cache_ptr));
  return SchedulerEntry::makeEntry(
      sg->heuristic(), local_fusion, runtime_info, data_cache);
}

std::unique_ptr<FusionHeuristics> SegmentedFusion::makeInitialHeuristics(
    const KernelArgumentHolder& inputs) {
  auto ret = std::make_unique<FusionHeuristics>();
  SchedulerRuntimeInfo runtime_info(completeFusion(), inputs, true);
  for (auto g : groups()) {
    ret->emplaceBack(makeInitialSchedulerEntry(g, runtime_info));
  }
  return ret;
}

HeuristicSummary* SegmentedFusion::getCachedHeuristicDataFor(
    SegmentedGroup* group) {
  auto data_it = heuristic_summary_cache_.find(group);
  if (data_it == heuristic_summary_cache_.end()) {
    return nullptr;
  }
  return data_it->second.get();
}

void SegmentedFusion::setCachedHeuristicDataFor(
    SegmentedGroup* group,
    std::unique_ptr<HeuristicSummary> data) {
  TORCH_INTERNAL_ASSERT(!heuristic_summary_cache_.count(group));
  heuristic_summary_cache_[group] = std::move(data);
}

namespace {

//! A thin traversal class that collects all the tensorviews
//!  that could cast to fp16 or bf16 if they were segmented edges.
//!  The selected values are currently defined as all the
//!  tensorviews that
//!     1. are not complete fusion input/output,
//!     2. have a use chain that ends with a fp16
//!         complete fusion output
//!     3. are fp32 datatype
class ForceHalfAnnotation : public IterVisitor {
 public:
  static std::unordered_set<TensorView*> getFP16AnnotatedSet(Fusion* fusion) {
    ForceHalfAnnotation annotation;
    std::vector<Val*> fp16_outputs;
    auto& cast_to_type = annotation.cast_to_type_;
    auto other_half_type =
        cast_to_type == DataType::Half ? DataType::BFloat16 : DataType::Half;
    std::copy_if(
        fusion->outputs().begin(),
        fusion->outputs().end(),
        std::back_inserter(fp16_outputs),
        [&cast_to_type, &other_half_type](auto* val) {
          auto dtype = val->getDataType().value();
          if (cast_to_type) {
            TORCH_INTERNAL_ASSERT(
                other_half_type != dtype,
                "Mix of BFloat16 and Float16 in the same graph is not supported.");
          }
          return val->template isA<TensorView>() &&
              val->getDataType().has_value() &&
              (val->getDataType().value() == DataType::Half ||
               val->getDataType().value() == DataType::BFloat16);
        });

    annotation.traverseTo(fusion, fp16_outputs);
    return annotation.force_fp16_tv_set_;
  }

 private:
  using IterVisitor::handle;

  void handle(TensorView* tv) override {
    auto dtype = tv->getDataType();
    if (dtype.has_value() && dtype.value() == DataType::Float &&
        !tv->isFusionOutput() && !tv->isFusionInput()) {
      force_fp16_tv_set_.insert(tv);
    }
  }

  std::unordered_set<TensorView*> force_fp16_tv_set_;
  c10::optional<DataType> cast_to_type_ = c10::nullopt;
};

} // namespace

void SegmentedFusion::annotateFP16IntermediateTensors() {
  force_fp16_tv_set_ =
      ForceHalfAnnotation::getFP16AnnotatedSet(complete_fusion_.get());
  for (auto out_tv :
       ir_utils::filterByType<TensorView>(complete_fusion_->outputs())) {
    if (out_tv) {
      auto dtype = out_tv->getDataType().value();
      if (dtype == DataType::Half || dtype == DataType::BFloat16) {
        force_half_precision_type_ = dtype;
      }
    }
  }
}

std::string toString(const SegmentCandidateFinderOptions& segment_options) {
  std::stringstream ss;
  ss << "segmentation phases {\n";
  if (segment_options.run_combine_reductions) {
    ss << "combine reductions\n";
  }
  if (segment_options.run_herrmann_merge) {
    ss << "herrmann merging\n";
  }
  if (segment_options.run_final_merge) {
    ss << "final merging\n";
  }
  ss << "\n}\n";
  return ss.str();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
