#include <scheduler/transpose.h>

#include <executor_utils.h>
#include <inlining.h>
#include <instrumentation.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <lower_utils.h>
#include <scheduler/pointwise_utils.h>
#include <scheduler/registry.h>
#include <scheduler/utils.h>
#include <scheduler/vectorize_helper.h>
#include <transform_replay.h>
#include <utils.h>

#include <ATen/cuda/CUDAContext.h>

#include <algorithm>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// DomainMap uses the ComputeAtMap to find a reference TensorView
// that maps to all iterDomains in the fusion.
class DomainMap : public pointwise_utils::DomainMap {
 public:
  using pointwise_utils::DomainMap::DomainMap;

  TensorView* findReferenceFor(const std::vector<TensorView*>& group) const {
    TensorView* result = nullptr;
    int max_dims = -1;
    for (auto tv : group) {
      if (isValidReference(tv)) {
        int dims = pointwise_utils::nRootDims(tv);
        if (dims > max_dims) {
          result = tv;
          max_dims = dims;
        }
      }
    }
    return result;
  }

  IterDomain* getMappedRootDimIn(TensorView* tv, IterDomain* root_dim) const {
    // Find the root id mapped to `root_dim`
    const auto& root_dom = tv->getRootDomain();
    IterDomain* mapped_id = nullptr;
    for (auto i : c10::irange(root_dom.size())) {
      if (ca_map_.idGraph().permissiveNodes().permissiveAreMapped(
              root_dom[i], root_dim)) {
        mapped_id = root_dom[i];
        break;
      }
    }
    return mapped_id;
  }

  static bool hasAtLeastTwoValidGroups(Fusion* fusion) {
    FusionGuard fg(fusion);
    DomainMap domain_map(fusion);
    auto grouped_inputs_outputs = domain_map.groupInputsOutputsByInnerDim();
    if (grouped_inputs_outputs.size() < 2) {
      return false;
    }
    auto ref1 = domain_map.findReferenceFor(grouped_inputs_outputs[0]);
    auto ref2 = domain_map.findReferenceFor(grouped_inputs_outputs[1]);
    if (ref1 == nullptr || ref2 == nullptr) {
      return false;
    }
    // reference 1 is the global reference, so it must have dim mapped the
    // innermost dim of both groups
    auto innermost2 = scheduler_utils::innerMostRootDim(ref2);
    return domain_map.getMappedRootDimIn(ref1, innermost2) != nullptr;
  }

  int getInnerLeafDim(TensorView* tv, IterDomain* root_dim) const {
    auto mapped_id = getMappedRootDimIn(tv, root_dim);
    TORCH_INTERNAL_ASSERT(
        mapped_id != nullptr,
        "Can not find ID mapped to ",
        root_dim,
        " in tensor ",
        tv);
    // Project the root id to leaf id
    while (!mapped_id->uses().empty()) {
      TORCH_INTERNAL_ASSERT(mapped_id->uses().size() == 1);
      auto expr = mapped_id->uses()[0];
      if (expr->isA<Split>()) {
        mapped_id = expr->as<Split>()->inner();
      } else {
        auto merge = expr->as<Merge>();
        TORCH_INTERNAL_ASSERT(
            mapped_id == merge->inner(),
            "Can not find ID mapped to ",
            root_dim,
            " in tensor ",
            tv);
        mapped_id = merge->out();
      }
    }
    // Find the position of the leaf id
    const auto& dom = tv->domain()->domain();
    for (auto i : c10::irange(dom.size())) {
      if (dom[i] == mapped_id) {
        return i;
      }
    }
    TORCH_INTERNAL_ASSERT(
        false, "Can not find ID mapped to ", root_dim, " in tensor ", tv);
  }

  // Group inputs and outputs of a fusion by its inner most domain. For example
  //   inputs: t0, t1
  //   t2 = transpose(t1)
  //   t3 = t0 + t2
  //   t4 = sin(t0)
  //   t5 = cos(t1)
  //   outputs: t3, t4, t5
  //
  // Then we should have group {t0, t3, t4} and {t1, t5}
  //
  // The returned groups are sorted in descending size. If the sizes of two
  // group are equal, then we sort them by their members in the following order:
  //   output[0], output[1], ..., input[0], input[1], ...
  // That is, {ouput[0], output[2]} will be in front of {ouput[1], output[3]}
  // The order here must be deterministic, because in transpose heuristics, we
  // have `vectorize_factor1` and `vectorize_factor2` and we need to be sure
  // that `1` and `2` are assigned to the same group across runs.
  //
  // In the case where view is present in the graph, there are two cases: if the
  // view doesn't touch any inner dimension of any group, then the support of it
  // is trivial. In the case where view actually touches an inner-most dim, we
  // keep track of the inner-most dimension of view's split and merges.
  //
  // For example, if you have:
  //   T0 [2, 3, 5] <-- input
  //   T1 [2, 5, 3] <-- input
  //   T2 [2, 5, 3] = transpose(T0) + T1
  //   T3 [2, 15] = view(T2)
  //   output <-- T3
  //
  // Then T3 should be in the same group with T1, and T0 should have
  // different group with T1 and T3.
  std::vector<std::vector<TensorView*>> groupInputsOutputsByInnerDim() const {
    std::vector<std::vector<TensorView*>> groups;
    auto output_tvs = ir_utils::filterByType<TensorView>(fusion_->outputs());
    auto input_tvs = ir_utils::filterByType<TensorView>(fusion_->inputs());
    std::unordered_set<TensorView*> grouped;
    decltype(input_tvs)* tv_filtered_groups[2] = {&output_tvs, &input_tvs};
    for (auto tv_filtered_group : tv_filtered_groups) {
      for (auto tv : *tv_filtered_group) {
        if (tv->isFusionInput() && tv->uses().empty()) {
          continue;
        }
        if (grouped.count(tv) > 0) {
          continue;
        }
        groups.emplace_back(std::vector<TensorView*>{tv});
        grouped.emplace(tv);
        // We only want to grab the inner-most dimension, because we don't want
        // tensors with different inner-most dimension to be put in the same
        // group. For example, if we have:
        //   T2[i1, i3*i2] = relu(view(transpose(T1[i1, i2, i3])))
        // then we don't want T1 and T2 to be in the same group.
        //
        // But we don't want to check contiguity. For example, if we have:
        //   T1[i1, i2, i3] (contiguous) + T2[i1, i2, i3] (discontiguous)
        // Then we still want to T1 and T2 to be grouped together.
        auto group =
            scheduler_utils::getInputsOutputsWithInnerDim(tv, true, false);
        if (group.empty()) {
          // In case that the inner most dim of tv is not found (for example, tv
          // is a fusion input with only reductions), we just return a null
          // result which will tell the scheduler to reject the fusion
          return {};
        }
        for (auto member_tv : group) {
          if (grouped.count(member_tv) == 0) {
            grouped.emplace(member_tv);
            groups.back().emplace_back(member_tv);
          } else if (member_tv != tv) {
            // Ambiguous grouping. This should only happen at `canSchedule`, so
            // we just return a null result which will tell the scheduler to
            // reject the fusion
            return {};
          }
        }
      }
    }
    std::stable_sort(
        groups.begin(),
        groups.end(),
        [](const std::vector<TensorView*>& v1,
           const std::vector<TensorView*>& v2) {
          return v1.size() > v2.size();
        });
    return groups;
  }
};

// Note: [Supporting small transpose dimensions]
// We prefer to make tiles of size 32x32 if there are enough elements to achieve
// good occupancy, otherwise, we will use tile size 8x8. In both cases, it is
// possible that the inner dimension of group 1 and/or group 2 are smaller than
// the desired tile size. If this happens, part of the threads of a block will
// be wasted, leading to bad performance. To prevent this from happening, if the
// size of the inner-most dim is smaller than the tile size, we merge other
// dimensions with the inner-most dimension to create larger "virtual inner-most
// dimension". The algorithm that we create these virtual inner-most dimensions
// is as follows:
//
// For example, if we have
//   T0[I0{2}, I1{1024*1024}, I2{2}, I3{2}, I4{2}, I5{2}, I6{2}] input
//   T1 = transpose(T0, 4, 6)
// We first try to merge each inner-most dim with the dimensions on its left:
//   T0[I0{2}, I1*I2*I3*I4{1024*1024*8}, I5*I6{4}]
// If there is/are still unsatisfied innermost dim(s) after this step (I5*I6 in
// this example), we find other dims that is not merged yet to satisfy it/them:
//   T0[I0*I5*I6{8}, I1*I2*I3*I4{1024*1024*8}]
// If after merging all the dims, there is still one of them not satisfied, this
// usually means there is one large dim that is consumed by the satisfied one.
// We will split that dim and large dim and and use the splitted ones to satisfy
// both of them:
//   T0[I0*I1o*I5*I6{1024*1024/4*8}, I1i*I2*I3*I4{32}]
void maybeBuildVirtualInnerDims(
    TransposeParams& params,
    int64_t device_multiprocessor_count,
    int64_t n_elems,
    const std::vector<int64_t>& shape_in_ref1,
    int64_t inner_most1,
    int64_t inner_most2) {
  int64_t merged_size1 = shape_in_ref1[inner_most1];
  int64_t merged_size2 = shape_in_ref1[inner_most2];

  int64_t actual_tile_size1 =
      std::min<int64_t>(merged_size1, params.tile_size1);
  int64_t actual_tile_size2 =
      std::min<int64_t>(merged_size2, params.tile_size2);
  int64_t wave_elements =
      device_multiprocessor_count * actual_tile_size1 * actual_tile_size2;

  if (wave_elements >= n_elems) {
    // if one full wave can handle all elements, don't create virtual inner dims
    return;
  }

  // merge inner_most1 and inner_most2 left until we are done or we can no
  // longer do so
  int64_t dim = inner_most1 - 1;
  while (dim >= 0 && dim != inner_most2 &&
         merged_size1 < (int64_t)params.tile_size1) {
    params.dims_merged_with_1.push_back(dim);
    merged_size1 *= shape_in_ref1[dim];
    dim--;
  }
  dim = inner_most2 - 1;
  while (dim >= 0 && dim != inner_most1 &&
         merged_size2 < (int64_t)params.tile_size2) {
    params.dims_merged_with_2.push_back(dim);
    merged_size2 *= shape_in_ref1[dim];
    dim--;
  }
  // If any of them are unsatisfied, then find other dims to merge
  std::unordered_set<int64_t> unavailable_dims;
  unavailable_dims.reserve(
      2 + params.dims_merged_with_1.size() + params.dims_merged_with_2.size());
  unavailable_dims.insert(inner_most1);
  unavailable_dims.insert(inner_most2);
  for (auto i : params.dims_merged_with_1) {
    unavailable_dims.insert(i);
  }
  for (auto i : params.dims_merged_with_2) {
    unavailable_dims.insert(i);
  }
  dim = shape_in_ref1.size() - 1;
  while (dim >= 0 && merged_size1 < (int64_t)params.tile_size1) {
    if (unavailable_dims.count(dim) == 0) {
      params.dims_merged_with_1.push_back(dim);
      merged_size1 *= shape_in_ref1[dim];
      unavailable_dims.insert(dim);
    }
    dim--;
  }
  dim = shape_in_ref1.size() - 1;
  while (dim >= 0 && merged_size2 < (int64_t)params.tile_size2) {
    if (unavailable_dims.count(dim) == 0) {
      params.dims_merged_with_2.push_back(dim);
      merged_size2 *= shape_in_ref1[dim];
      unavailable_dims.insert(dim);
    }
    dim--;
  }
  // If both are satisfied, then we are done. If neither are satisfied, then it
  // is impossible to satisfy both of them, also done.
  if ((merged_size1 < (int64_t)params.tile_size1) ==
      (merged_size2 < (int64_t)params.tile_size2)) {
    return; // no need to split
  }
  // If one of them are not satisfied, there might be two cases:
  // 1. The satisfied one just merged in a large dim. If this is the case, We
  //    split this large dim, so that now we have two available dims to satisfy
  //    both virtual innermost dim.
  // 2. The satisfied one did not merge in anything. For example,
  //    T0[I0{1024*1024}, I1{2}]
  //    If this is the case, this means that we need to split the large
  //    inner-most dimension to satisfy the small innermost dimension
  int64_t large_dim;
  int64_t split_factor;
  bool split_inner_most;
  if (merged_size1 < (int64_t)params.tile_size1) {
    if (params.dims_merged_with_2.empty()) {
#if SUPPORT_SPLITTING_INNERMOST_DIM
      // https://github.com/csarofeen/pytorch/issues/1964
      // case 2
      split_inner_most = true;
      large_dim = inner_most2;
      split_factor = params.tile_size2;
#else
      // disabled due to indexing error
      return;
#endif
    } else {
      // case 1
      split_inner_most = false;
      large_dim = params.dims_merged_with_2.back();
      auto prev_merged_size2 = merged_size2 / shape_in_ref1[large_dim];
      split_factor = ceilDiv(params.tile_size2, prev_merged_size2);
    }
  } else {
    if (params.dims_merged_with_1.empty()) {
#if SUPPORT_SPLITTING_INNERMOST_DIM
      // https://github.com/csarofeen/pytorch/issues/1964
      // case 2
      split_inner_most = true;
      large_dim = inner_most1;
      split_factor = params.tile_size1;
#else
      // disabled due to indexing error
      return;
#endif
    } else {
      // case 1
      split_inner_most = false;
      large_dim = params.dims_merged_with_1.back();
      auto prev_merged_size1 = merged_size1 / shape_in_ref1[large_dim];
      split_factor = ceilDiv(params.tile_size1, prev_merged_size1);
    }
  }
  params.split_before_tiling.push_back({large_dim, split_factor});
  // adjust all dims to after-split
  for (auto& i : params.dims_merged_with_1) {
    if ((int64_t)i > large_dim) {
      i++;
    }
  }
  for (auto& i : params.dims_merged_with_2) {
    if ((int64_t)i > large_dim) {
      i++;
    }
  }
  // Give the split-out dim to the unsatisfied one, so that both are satisfied.
  if (merged_size1 < (int64_t)params.tile_size1) {
    if (!split_inner_most) {
      params.dims_merged_with_2.pop_back();
      params.dims_merged_with_2.push_back(large_dim + 1);
    }
    params.dims_merged_with_1.push_back(large_dim);
  } else {
    if (!split_inner_most) {
      params.dims_merged_with_1.pop_back();
      params.dims_merged_with_1.push_back(large_dim + 1);
    }
    params.dims_merged_with_2.push_back(large_dim);
  }
}

HeuristicSummaryEntry<HeuristicCompileTime::TransposeDomainMap> getDomainMap(
    HeuristicSummary* data_cache,
    Fusion* fusion) {
  auto domain_map_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::TransposeDomainMap>(
          data_cache,
          [fusion]() { return std::make_unique<DomainMap>(fusion); });
  return domain_map_entry;
}

HeuristicSummaryEntry<HeuristicCompileTime::InputsOutputsInnerDimGroups>
getInputsOutputsGroups(HeuristicSummary* data_cache, DomainMap& domain_map) {
  auto grouped_inputs_outputs_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::InputsOutputsInnerDimGroups>(
          data_cache, [&domain_map]() {
            return std::make_unique<std::vector<std::vector<TensorView*>>>(
                domain_map.groupInputsOutputsByInnerDim());
          });
  auto& grouped_inputs_outputs = grouped_inputs_outputs_entry.get();

  TORCH_INTERNAL_ASSERT(
      grouped_inputs_outputs.size() >= 2,
      "Can not find mismatched inner most dim, should use pointwise scheduler.");

  return grouped_inputs_outputs_entry;
}

HeuristicSummaryEntry<HeuristicCompileTime::ReferenceTensorsForGroups>
getReferenceTensors(
    HeuristicSummary* data_cache,
    DomainMap& domain_map,
    std::vector<std::vector<TensorView*>>& grouped_inputs_outputs) {
  auto reference_tensors_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::ReferenceTensorsForGroups>(
          data_cache, [&domain_map, &grouped_inputs_outputs]() {
            std::vector<TensorView*> data{
                domain_map.findReferenceFor(grouped_inputs_outputs[0]),
                domain_map.findReferenceFor(grouped_inputs_outputs[1])};
            return std::make_unique<std::vector<TensorView*>>(std::move(data));
          });
  auto& reference_tensors = reference_tensors_entry.get();
  TORCH_INTERNAL_ASSERT(reference_tensors.size() == 2);
  TensorView* reference1 = reference_tensors[0];
  TensorView* reference2 = reference_tensors[1];
  TORCH_INTERNAL_ASSERT(
      reference1 != nullptr, "Unable to find reference tensor for group 1");
  TORCH_INTERNAL_ASSERT(
      reference2 != nullptr, "Unable to find reference tensor for group 2");
  return reference_tensors_entry;
}

std::pair<std::vector<int64_t>, int64_t> getShapeInReference(
    HeuristicSummary* data_cache,
    SchedulerRuntimeInfo& runtime_info,
    TensorView* reference,
    DomainMap& domain_map) {
  auto ref_root = reference->getMaybeRFactorDomain();
  std::vector<int64_t> shape_in_ref;
  shape_in_ref.reserve(reference->nDims());
  int64_t n_elems = 1;
  for (size_t ref_i = 0; ref_i < ref_root.size(); ref_i++) {
    auto id = ref_root[ref_i];
    auto concrete_id = domain_map.getComputeAtMap().getConcreteMappedID(
        id, IdMappingMode::EXACT);
    auto inferred_val =
        runtime_info.expressionEvaluator().evaluate(concrete_id->extent());
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(),
        "Error inferring size for pointwise scheduler: ",
        ref_root[ref_i]->extent()->toInlineString());
    int64_t size = inferred_val->as<int64_t>();
    n_elems *= size;
    shape_in_ref.push_back(size);
  }
  return {shape_in_ref, n_elems};
}

HeuristicSummaryEntry<HeuristicCompileTime::InnerMostDimInfo>
getInnerMostDimInfoInReference(
    HeuristicSummary* data_cache,
    const std::vector<TensorView*>& group_references,
    TensorView* global_reference,
    DomainMap& domain_map) {
  auto innermost_info_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::InnerMostDimInfo>(
          data_cache, [&]() {
            std::vector<int64_t> data;
            data.reserve(group_references.size());
            for (auto ref_tv : group_references) {
              auto inner_most_id = scheduler_utils::innerMostRootDim(ref_tv);
              auto inner_most_pos_in_global_ref =
                  domain_map.getInnerLeafDim(global_reference, inner_most_id);
              data.emplace_back(inner_most_pos_in_global_ref);
            }
            return std::make_unique<std::vector<int64_t>>(std::move(data));
          });
  return innermost_info_entry;
}

} // namespace

std::string getTransposeRuntimeRejectReason(
    Fusion* fusion,
    HeuristicSummary* data_cache,
    SchedulerRuntimeInfo& runtime_info) {
  auto domain_map_entry = getDomainMap(data_cache, fusion);
  auto& domain_map = dynamic_cast<DomainMap&>(domain_map_entry.get());
  auto grouped_inputs_outputs_entry =
      getInputsOutputsGroups(data_cache, domain_map);
  auto grouped_inputs_outputs = grouped_inputs_outputs_entry.get();
  auto reference_tensors_entry =
      getReferenceTensors(data_cache, domain_map, grouped_inputs_outputs);
  auto reference_tensors = reference_tensors_entry.get();
  TensorView* reference1 = reference_tensors[0];

  auto pair =
      getShapeInReference(data_cache, runtime_info, reference1, domain_map);
  auto& shape_in_ref1 = pair.first;
  auto& n_elems = pair.second;

  auto innermost_info_entry = getInnerMostDimInfoInReference(
      data_cache, reference_tensors, reference1, domain_map);
  auto innermost_info = innermost_info_entry.get();

  constexpr size_t default_tile_elements =
      TransposeParams::getDefaultTileSize() *
      TransposeParams::getDefaultTileSize();

  // don't schedule with transpose scheduler if less than a full wave
  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  auto elements_per_wave = device_multiprocessor_count * default_tile_elements;
  if ((int64_t)elements_per_wave > n_elems) {
    return "Transpose scheduler does not perform well on small problem sizes.";
  }

  auto inner_most_pos1_in_ref1 = innermost_info[0];
  auto inner_most_pos2_in_ref1 = innermost_info[1];

  auto inner_size1 = shape_in_ref1[inner_most_pos1_in_ref1];
  auto inner_size2 = shape_in_ref1[inner_most_pos2_in_ref1];

  // For cases like
  //   transpose(T0[1000000000, 2, 2], 1, 2)
  // the pointwise scheduler should provide better performance, because it
  // provides coalesced memory access
  if (inner_size1 * inner_size2 < (int64_t)default_tile_elements) {
    auto inner_elements = inner_size1 * inner_size2;
    for (int64_t i = inner_most_pos2_in_ref1 + 1; i < inner_most_pos1_in_ref1;
         i++) {
      inner_elements *= shape_in_ref1[i];
    }
    // note that the algorithm here is only an approximation because it only
    // checks reference1. In principle, we need to check all inputs and outputs
    // to get an accurate result, but that is too much work. I think checking
    // only reference 1 is fine for now. Below is an example where the
    // approximation here will not work:
    //   T0[10000000, 2, 3] (reference 1)
    //   T1[2, 10000000, 3] input/output
    //   T2[2, 10000000, 3] input/output
    //   T3[2, 10000000, 3] input/output
    //   T4[3, 10000000, 2] input/output
    //   T5[3, 10000000, 2] input/output
    if (inner_elements < (int64_t)default_tile_elements) {
      return "Inner transpose of small dimensions should be scheduled by the "
             "pointwise scheduler because it provides better memory coalescing";
    }
  }

#if !SUPPORT_SPLITTING_INNERMOST_DIM
  if (n_elems / inner_size1 < (int64_t)TransposeParams::getDefaultTileSize() ||
      n_elems / inner_size2 < (int64_t)TransposeParams::getDefaultTileSize()) {
    return "Splitting of inner most dim for the creation of virtual inner most dim "
           "is disabled due to indexing bug, skipping this case at runtime for now"
           "See: https://github.com/csarofeen/pytorch/issues/1964";
  }
#endif

  return "";
}

bool hasAtLeastTwoValidGroups(Fusion* fusion) {
  return DomainMap::hasAtLeastTwoValidGroups(fusion);
}

std::shared_ptr<TransposeParams> getTransposeHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache) {
  SchedulerRuntimeInfo runtime_info(fusion, runtime_inputs, true);
  return getTransposeHeuristics(fusion, runtime_info, data_cache);
}

std::shared_ptr<TransposeParams> getTransposeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("getTransposeHeuristics");

  FusionGuard fg(fusion);

  // Incase any buffer is of type DataType::Index
  DataType index_type = indexModeToDtype(runtime_info.getIndexMode());

  auto domain_map_entry = getDomainMap(data_cache, fusion);
  auto& domain_map = dynamic_cast<DomainMap&>(domain_map_entry.get());
  auto grouped_inputs_outputs_entry =
      getInputsOutputsGroups(data_cache, domain_map);
  auto grouped_inputs_outputs = grouped_inputs_outputs_entry.get();
  auto reference_tensors_entry =
      getReferenceTensors(data_cache, domain_map, grouped_inputs_outputs);
  auto reference_tensors = reference_tensors_entry.get();
  TensorView* reference1 = reference_tensors[0];
  TensorView* reference2 = reference_tensors[1];
  auto pair =
      getShapeInReference(data_cache, runtime_info, reference1, domain_map);
  auto& shape_in_ref1 = pair.first;
  auto& n_elems = pair.second;

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  auto innermost_info_entry = getInnerMostDimInfoInReference(
      data_cache, reference_tensors, reference1, domain_map);
  auto innermost_info = innermost_info_entry.get();

  auto inner_most_pos1_in_ref1 = innermost_info[0];
  auto inner_most_pos2_in_ref1 = innermost_info[1];

  auto params = std::make_shared<TransposeParams>("Transpose heuristics");

  // Expand inner-most dims to virtual inner-most dims so that the inner-most
  // dims has at least tile_size elements
  // See note [Supporting small transpose dimensions]
  maybeBuildVirtualInnerDims(
      *params,
      device_multiprocessor_count,
      n_elems,
      shape_in_ref1,
      inner_most_pos1_in_ref1,
      inner_most_pos2_in_ref1);

  // Note [vectorization and unroll of input and output]
  //
  // The choice of vectorization size, block size and tile sizes needs to be
  // consistent with each other. Consider the following:
  //
  // The number of threads in one block is
  //   num_threads = blockDim.x * blockDim.y
  // and the number of elements per each tile is
  //   num_elems_per_tile = params->tile_size1 * params->tile_size2
  // So each thread needs to process
  //   num_elems_per_thread = num_elems_per_tile / num_threads
  // elements. That is, once the tile sizes and block size are determined, the
  // `num_elems_per_thread` is determined, regardless of vectorizability of
  // input/output tensors.
  //
  // To make the selection of tile sizes othogonal to vectorizability, we
  // support having both vectorization and unrolling in the same tensor. For
  // example, if we have num_elems_per_tile == 1024 and num_threads = 256, then
  // we have num_elems_per_thread being 4. And if we have vector size 2, then we
  // will do unroll 2 * vectorize 2 at the same tensor.
  //
  // Also, since the inner most dim of different groups are not the same, it is
  // natural to consider their vectorizability separately and allow them to have
  // different vectorize/unroll sizes.

  constexpr int64_t kSixteen = 16; // clang tidy

  int64_t max_input_dtype_size = 1;

  size_t n_input_tensors = 0;
  for (auto inp : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    max_input_dtype_size = std::max(
        max_input_dtype_size,
        (int64_t)dataTypeSize(inp->getDataType().value(), index_type));
    n_input_tensors++;
  }

  auto max_unroll_factor = ceilDiv(
      // Available unrolling based on size of data type
      (int64_t)kSixteen / max_input_dtype_size,
      // Reduce max unrolling factor if we have many inputs/outputs to unroll
      // as it could start consuming a lot of registers.
      std::max(
          (scheduler_utils::lastPow2(
               (int64_t)grouped_inputs_outputs[0].size() +
               (int64_t)grouped_inputs_outputs[1].size()) >>
           2),
          (int64_t)1));

  // Don't unroll at the cost of getting a full wave on the GPU
  auto max_unroll_factor_occupancy = ceilDiv(
      n_elems,
      device_multiprocessor_count * params->tile_size1 * params->tile_size2);
  max_unroll_factor = std::min(max_unroll_factor, max_unroll_factor_occupancy);

  // Don't unroll at the cost of getting a full warp, useful for the case where
  // tile sizes are small
  auto max_unroll_factor_block =
      ceilDiv(params->tile_size1 * params->tile_size2, 32);
  max_unroll_factor = std::min(max_unroll_factor, max_unroll_factor_block);

  // Compute maximum vectorize factor that can be used
  size_t vectorize_factor1 = max_unroll_factor;
  size_t vectorize_factor2 = max_unroll_factor;

  for (auto tv : grouped_inputs_outputs[0]) {
    const auto tv_vectorize_factor =
        runtime_info.getInnerDimVectorizableWidth(tv);
    vectorize_factor1 = std::min(vectorize_factor1, tv_vectorize_factor);
  }
  // TODO: Since group2 only has global->shared and shared->global set op, we
  // can have fine-grained control of unroll/vectorization at per tensor level.
  // We should not be using a single global vectorize factor for the entire
  // group 2
  for (auto tv : grouped_inputs_outputs[1]) {
    const auto tv_vectorize_factor =
        runtime_info.getInnerDimVectorizableWidth(tv);
    vectorize_factor2 = std::min(vectorize_factor2, tv_vectorize_factor);
  }

  params->vectorize_factor1 = scheduler_utils::lastPow2(
      std::min(static_cast<size_t>(max_unroll_factor), vectorize_factor1));
  params->vectorize_factor2 = scheduler_utils::lastPow2(
      std::min(static_cast<size_t>(max_unroll_factor), vectorize_factor2));

  params->lparams.bind(params->getThreadsPerBlock(), ParallelType::TIDx);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    std::cerr << "\n===== Transpose Stats ========\n"
              << "inputs: " << ir_utils::toString(fusion->inputs()) << "\n"
              << "outputs: " << ir_utils::toString(fusion->outputs()) << "\n"
              << "shape: " << shape_in_ref1 << "\n"
              << "num_elems: " << n_elems << "\n"
              << "n_input_tensors: " << n_input_tensors << "\n"
              << "max_input_dtype_size: " << max_input_dtype_size << "\n"
              << "group 1: " << ir_utils::toString(grouped_inputs_outputs[0])
              << "\n"
              << "reference1: " << reference1 << "\n"
              << "inner_most_id1 position: " << inner_most_pos1_in_ref1
              << " (in reference 1)\n"
              << "group 2: " << ir_utils::toString(grouped_inputs_outputs[1])
              << "\n"
              << "reference2: " << reference2 << "\n"
              << "inner_most_id2 position: " << inner_most_pos2_in_ref1
              << " (in reference 1)" << std::endl;
    if (!params->split_before_tiling.empty() ||
        !params->dims_merged_with_1.empty() ||
        !params->dims_merged_with_2.empty()) {
      std::cerr << "small transposed dim, needs virtual inner-most dim"
                << std::endl;
    }
    std::cerr << std::endl;
    std::cerr << params->toString() << std::endl;
  }

  return params;
}

// TODO: remove or return launch parameters
LaunchParams scheduleTranspose(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs) {
  FUSER_PERF_SCOPE("scheduleFusion");
  auto params = getTransposeHeuristics(fusion, runtime_inputs);
  TORCH_INTERNAL_ASSERT(
      params != nullptr, "Could not schedule transpose operation.");
  scheduleTranspose(fusion, *params);
  return params->lparams;
}

void scheduleTranspose(Fusion* fusion, TransposeParams params) {
  FusionGuard fg(fusion);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  // maybe has_reduction for scheduling should be done on a per output tensor
  // basis.
  // TODO: add support for trivial reduction
  TORCH_INTERNAL_ASSERT(
      ir_utils::getReductionOps(fusion, /*ignore_trivial=*/false).empty(),
      "This scheduler only handles pointwise ops.");

  // Cache inputs
  auto cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  auto cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, true);

  std::vector<TensorView*> input_tvs;
  {
    auto filtered_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    // Remove hanging tensor views
    for (auto tv : filtered_tvs) {
      if (tv->uses().empty()) {
        continue;
      }
      input_tvs.push_back(tv);
    }
  }
  auto output_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());

  size_t max_dims = 0;
  for (auto inp : input_tvs) {
    max_dims = std::max(pointwise_utils::nRootDims(inp), max_dims);
  }

  for (auto out : output_tvs) {
    max_dims = std::max(pointwise_utils::nRootDims(out), max_dims);
  }

  // If everything is zero dim tensors, just return.
  if (max_dims == 0) {
    return;
  }

  DomainMap domain_map(fusion);
  auto grouped_inputs_outputs = domain_map.groupInputsOutputsByInnerDim();
  TORCH_INTERNAL_ASSERT(grouped_inputs_outputs.size() >= 2);

  /*
   * We need something similar to `cacheFork` for input tensors in group 2. We
   * need this because we will want to propagate to the entire DAG except group
   * 2 and its cached inputs, so we need to make sure the DAG is still connected
   * if we remove group and its cached inputs. For example
   *    t0
   *    |
   *   cache
   *   /  \
   *  t1  t2
   * if groups = {{t1, t2}, {t0}}, then removing {t0, cache} from the DAG will
   * make it disconnected.
   */
  std::unordered_set<TensorView*> group2_and_cached_inputs(
      grouped_inputs_outputs[1].begin(), grouped_inputs_outputs[1].end());
  for (auto tv : grouped_inputs_outputs[1]) {
    if (tv->isFusionInput()) {
      auto existing_cache = ir_utils::consumerTvsOf(tv)[0];
      if (ir_utils::consumerTvsOf(existing_cache).size() > 1) {
        auto new_cache = tv->cacheAfter();
        new_cache->setMemoryType(MemoryType::Shared);
        group2_and_cached_inputs.emplace(new_cache);
      } else {
        existing_cache->setMemoryType(MemoryType::Shared);
        group2_and_cached_inputs.emplace(existing_cache);
      }
    }
  }
  // set cached outputs of group 2 to shared memory
  for (auto pair : cached_outputs) {
    auto cached_output = pair.first;
    auto output = pair.second;
    if (group2_and_cached_inputs.count(output) > 0) {
      cached_output->setMemoryType(MemoryType::Shared);
    }
  }

  TensorView* reference1 =
      domain_map.findReferenceFor(grouped_inputs_outputs[0]);
  TensorView* reference2 =
      domain_map.findReferenceFor(grouped_inputs_outputs[1]);

  TORCH_INTERNAL_ASSERT(
      reference1 != nullptr,
      "Could not find a fully broadcasted tensor to reference schedule on the first group.");

  TORCH_INTERNAL_ASSERT(
      reference2 != nullptr,
      "Could not find a fully broadcasted tensor to reference schedule on the second group.");

  auto inner_most_id1 = scheduler_utils::innerMostRootDim(reference1);
  auto inner_most_id2 = scheduler_utils::innerMostRootDim(reference2);

  //////////////////////////////////////////
  // Step 1: Make virtual inner most dims //
  //////////////////////////////////////////

  // See note [Supporting small transpose dimensions]

  // split big dims so that we have enough dimensions available to merge with
  // inner-most dims to create the virtual inner-most dim
  scheduler_utils::splitDims(reference1, params.split_before_tiling);
  // Merging reference 1's dims_merged_with_1 but updating dims_merged_with_2
  // based on the changes in the dimensions that were merged. So we can then run
  // merge with dims_merged_with_2.
  auto merged1 = scheduler_utils::mergeDims(
      reference1, params.dims_merged_with_1, params.dims_merged_with_2);
  // Merging reference 1's dims_merged_with_2 and updating `merged1`.
  std::vector<size_t> merged1_vec;
  if (merged1.has_value()) {
    merged1_vec.push_back(*merged1);
  }
  auto merged2 = scheduler_utils::mergeDims(
      reference1, params.dims_merged_with_2, merged1_vec);
  if (merged1.has_value()) {
    merged1 = merged1_vec[0];
  }

  // merge with inner most dims to get virtual inner most dims
  size_t inner_most_pos1_in_ref1 =
      domain_map.getInnerLeafDim(reference1, inner_most_id1);
  size_t inner_most_pos2_in_ref1 =
      domain_map.getInnerLeafDim(reference1, inner_most_id2);
  if (merged1.has_value()) {
    if (inner_most_pos1_in_ref1 < *merged1) {
      reference1->reorder(
          {{*merged1, inner_most_pos1_in_ref1},
           {inner_most_pos1_in_ref1, *merged1}});
      std::swap(*merged1, inner_most_pos1_in_ref1);
    }
    if (inner_most_pos2_in_ref1 > inner_most_pos1_in_ref1) {
      inner_most_pos2_in_ref1--;
    }
    if (merged2.has_value() && *merged2 > inner_most_pos1_in_ref1) {
      (*merged2)--;
    }
    reference1->merge(*merged1, inner_most_pos1_in_ref1);
    inner_most_pos1_in_ref1 = *merged1;
  }
  if (merged2.has_value()) {
    if (inner_most_pos2_in_ref1 < *merged2) {
      reference1->reorder(
          {{*merged2, inner_most_pos2_in_ref1},
           {inner_most_pos2_in_ref1, *merged2}});
      std::swap(*merged2, inner_most_pos2_in_ref1);
    }
    if (inner_most_pos1_in_ref1 > inner_most_pos2_in_ref1) {
      inner_most_pos1_in_ref1--;
    }
    reference1->merge(*merged2, inner_most_pos2_in_ref1);
    inner_most_pos2_in_ref1 = *merged2;
  }

  /////////////////////////////
  // Step 2: global schedule //
  /////////////////////////////

  // make tile
  // [..., I1, .., I2, ...]
  reference1->split(inner_most_pos1_in_ref1, params.tile_size1);
  reference1->reorder({{inner_most_pos1_in_ref1 + 1, -1}});
  reference1->split(inner_most_pos2_in_ref1, params.tile_size2);
  reference1->reorder({{inner_most_pos2_in_ref1 + 1, -1}});
  // [..., I1/tile1, .., I2/tile2, ..., tile1, tile2]

  // Merge remaining dimensions
  int lhs_i = -1;
  for (int i = (int)reference1->nDims() - 2; i > 0; i--) {
    auto axis_i = i - 1;
    if (lhs_i == -1) {
      lhs_i = axis_i;
    } else {
      reference1->merge(axis_i, lhs_i);
      lhs_i = axis_i;
    }
  }
  reference1->split(0, 1);
  // [merged_dim, 1, tile1, tile2]

  // parallelize non-tile dimensions
  reference1->axis(1)->parallelize(ParallelType::Unswitch);
  reference1->axis(0)->parallelize(ParallelType::BIDx);
  // [BIDx, Unswitch, tile1, tile2]

  // Propagate transformations so far to the entire DAG
  TransformPropagator propagator(reference1);
  MaxRootDomainInfoSpanningTree entire_dag(reference1);
  entire_dag.traverse(&propagator);
  scheduler_utils::parallelizeAllLike(reference1);

  // For a transpose scheduling, all we need is to bind threadIdx.x differently
  // for inputs and outputs. This swap of binding could happen at any tensor on
  // the path from input to output, especially, it does not have to be in the
  // transpose tensor. Here, we naively do the binding swap at cached
  // input/output for simplicity. We might need to find a better set of swap
  // tensors in the future to reduce shared memory usage.

  //////////////////////////////
  // Step 3: Schedule group 2 //
  //////////////////////////////

  // transform tile for vectorization/unroll
  // See note [vectorization and unroll of input and output]

  int pos = reference2->nDims() - 2;
  // [..., tile1, tile2]
  reference2->merge(pos);
  reference2->split(pos, params.vectorize_factor2);
  reference2->split(pos, params.getThreadsPerBlock());
  // [..., Unroll, TIDx, Vectorize]

  // Propagate transformations of reference2 to the entire DAG except
  // group 1. We actually only want to propagate to the fusion outputs, but
  // inputs and outputs themselves are disconnected, so we have to borrow the
  // entire DAG and use its spanning tree.
  {
    auto all_tvs_except1 = ir_utils::allTvsExcept(
        fusion,
        {grouped_inputs_outputs[0].begin(), grouped_inputs_outputs[0].end()});
    SetSelector selector({all_tvs_except1.begin(), all_tvs_except1.end()});
    MaxRootDomainInfoSpanningTree entire_dag_except1(reference2, &selector);
    TransformPropagator propagator(reference2);
    entire_dag_except1.traverse(&propagator);
  }

  // parallelize group2 and its cached inputs
  {
    if (params.vectorize_factor2 > 1) {
      reference2->axis(-1)->parallelize(ParallelType::Vectorize);
    }
    reference2->axis(-2)->parallelize(ParallelType::TIDx);
    reference2->axis(-3)->parallelize(ParallelType::Unroll);

    ComputeAtMap ca_map(fusion);

    scheduler_utils::parallelizeAllLike(
        reference2,
        {group2_and_cached_inputs.begin(), group2_and_cached_inputs.end()},
        {ParallelType::TIDx});

    // Only vectorize the axes that exactly maps to the vectorized axes
    //  on reference as support for permissively mapped axes are not
    //  yet clearly defined.
    std::vector<TensorView*> vectorized_group2_cached_inputs;
    for (auto gin : group2_and_cached_inputs) {
      if (std::any_of(
              gin->domain()->domain().begin(),
              gin->domain()->domain().end(),
              [&ca_map, reference2](IterDomain* id) {
                return ca_map.areMapped(
                    id, reference2->axis(-1), IdMappingMode::EXACT);
              })) {
        vectorized_group2_cached_inputs.push_back(gin);
      }
    }
    scheduler_utils::parallelizeAllLike(
        reference2, vectorized_group2_cached_inputs, {ParallelType::Vectorize});

    // Only unroll the axes that exactly maps to the unrolled axes
    //  on reference as support for permissively mapped axes are not
    //  yet clearly defined.
    std::vector<TensorView*> unrolled_group2_cached_inputs;
    for (auto gin : group2_and_cached_inputs) {
      if (std::any_of(
              gin->domain()->domain().begin(),
              gin->domain()->domain().end(),
              [&ca_map, reference2](IterDomain* id) {
                return ca_map.areMapped(
                    id, reference2->axis(-3), IdMappingMode::EXACT);
              })) {
        unrolled_group2_cached_inputs.push_back(gin);
      }
    }
    scheduler_utils::parallelizeAllLike(
        reference2, unrolled_group2_cached_inputs, {ParallelType::Unroll});
  }

  //////////////////////////////
  // Step 4: Schedule group 1 //
  //////////////////////////////

  // schedule group 1
  reference1->reorder({{-2, -1}});
  // [..., tile2, tile1]
  pos = reference1->nDims() - 2;
  reference1->merge(pos);
  reference1->split(pos, params.vectorize_factor1);
  reference1->split(pos, params.getThreadsPerBlock());
  if (params.vectorize_factor1 > 1) {
    reference1->axis(-1)->parallelize(ParallelType::Vectorize);
  }
  reference1->axis(-2)->parallelize(ParallelType::TIDx);
  reference1->axis(-3)->parallelize(ParallelType::Unroll);
  // [..., Unroll, TIDx, Vectorize]

  // Propagate transformations, parallelization of the reference1 to the entire
  // DAG except group 2 and its corresponding cached outputs.
  {
    auto all_tvs_except2 =
        ir_utils::allTvsExcept(fusion, group2_and_cached_inputs);
    SetSelector selector({all_tvs_except2.begin(), all_tvs_except2.end()});
    MaxRootDomainInfoSpanningTree entire_dag_except_outputs(
        reference1, &selector);
    TransformPropagator propagator(reference1);
    entire_dag_except_outputs.traverse(&propagator);
    scheduler_utils::parallelizeAllLike(
        reference1, all_tvs_except2, {ParallelType::TIDx});
  }

  // vectorize and unroll group 1's output and cached input
  {
    ComputeAtMap ca_map(fusion);
    std::vector<TensorView*> group1_and_cached_inputs(
        grouped_inputs_outputs[0].begin(), grouped_inputs_outputs[0].end());
    for (auto tv : grouped_inputs_outputs[0]) {
      if (tv->isFusionInput()) {
        group1_and_cached_inputs.emplace_back(ir_utils::consumerTvsOf(tv)[0]);
      }
    }

    // Only vectorize the axes that exactly maps to the vectorized axes
    //  on reference as support for permissively mapped axes are not
    //  yet clearly defined.
    std::vector<TensorView*> vectorized_group1_cached_inputs;
    for (auto gin : group1_and_cached_inputs) {
      if (std::any_of(
              gin->domain()->domain().begin(),
              gin->domain()->domain().end(),
              [&ca_map, reference1](IterDomain* id) {
                return ca_map.areMapped(
                    id, reference1->axis(-1), IdMappingMode::EXACT);
              })) {
        vectorized_group1_cached_inputs.push_back(gin);
      }
    }
    scheduler_utils::parallelizeAllLike(
        reference1, vectorized_group1_cached_inputs, {ParallelType::Vectorize});

    // Only unroll the axes that exactly maps to the unrolled axes
    //  on reference as support for permissively mapped axes are not
    //  yet clearly defined.
    std::vector<TensorView*> unrolled_group1_cached_inputs;
    for (auto gin : group1_and_cached_inputs) {
      if (std::any_of(
              gin->domain()->domain().begin(),
              gin->domain()->domain().end(),
              [&ca_map, reference1](IterDomain* id) {
                return ca_map.areMapped(
                    id, reference1->axis(-3), IdMappingMode::EXACT);
              })) {
        unrolled_group1_cached_inputs.push_back(gin);
      }
    }
    scheduler_utils::parallelizeAllLike(
        reference1, unrolled_group1_cached_inputs, {ParallelType::Unroll});
  }

  ////////////////////////////////
  // Step 5: Cleanup and inline //
  ////////////////////////////////

  // cleanup parallelization from reference1 and reference2 if they are fusion
  // inputs
  for (auto tv : {reference1, reference2}) {
    if (tv->isFusionInput()) {
      for (auto id : tv->domain()->domain()) {
        id->parallelize(ParallelType::Serial);
      }
    }
  }

  // Inline
  inlineMost();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
