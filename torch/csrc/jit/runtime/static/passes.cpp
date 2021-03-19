#include <torch/csrc/jit/runtime/static/passes.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {

void ConcatAddMulReplaceNaNClip(std::shared_ptr<torch::jit::Graph>& graph) {
  // TODO:: check restrictions for inputs; outputs not used elsewhere
  std::string pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g, %h, %i, %j):
        %y0 = aten::cat(%a, %b)
        %y1 = aten::add(%y0, %c, %d)
        %y2 = aten::mul(%y1, %e)
        %y3 = aten::nan_to_num(%y2, %f, %g, %h)
        %res = aten::clamp(%y3, %i, %j)
        return (%res))IR";
  std::string pattern2 = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g, %h, %i, %j):
        %y0 = aten::cat(%a, %b)
        %y1 = aten::add(%y0, %c, %d)
        %y2 = aten::mul(%y1, %e)
        %y3 = aten::nan_to_num_(%y2, %f, %g, %h)
        %res = aten::clamp(%y3, %i, %j)
        return (%res))IR";
  std::string pattern3 = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g, %h, %i, %j):
        %y0 = aten::cat(%a, %b)
        %y1 = aten::add(%y0, %c, %d)
        %y2 = aten::mul(%y1, %e)
        %y3 = aten::nan_to_num_(%y2, %f, %g, %h)
        %res = aten::clamp_(%y3, %i, %j)
        return (%res))IR";
  std::string pattern4 = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g, %h, %i, %j):
        %y0 = aten::cat(%a, %b)
        %y1 = aten::add(%y0, %c, %d)
        %y2 = aten::mul(%y1, %e)
        %y3 = aten::nan_to_num(%y2, %f, %g, %h)
        %res = aten::clamp_(%y3, %i, %j)
        return (%res))IR";
  std::string fused_pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g, %h, %i, %j):
        %res = fb::concat_add_mul_replacenan_clip(%c, %e, %a, %i, %j)
        return (%res))IR";

  SubgraphRewriter fuse;
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  fuse.runOnGraph(graph);

  fuse.RegisterRewritePattern(pattern2, fused_pattern);
  fuse.runOnGraph(graph);

  fuse.RegisterRewritePattern(pattern3, fused_pattern);
  fuse.runOnGraph(graph);

  fuse.RegisterRewritePattern(pattern4, fused_pattern);
  fuse.runOnGraph(graph);
}

void CastedBatchOneHotLengths(std::shared_ptr<torch::jit::Graph>& graph) {
  // TODO:: check restrictions for inputs; outputs not used elsewhere
  std::string pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g):
        %y0 : Tensor = aten::to(%a, %b, %c, %c, %d)
        %y1 : Tensor = fb::batch_one_hot_lengths(%y0, %e, %f)
        %res : Tensor = aten::to(%y1, %g, %c, %c, %d)
        return (%res))IR";
  std::string fused_pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g):
        %res : Tensor = fb::casted_batch_one_hot_lengths(%a, %e, %f)
        return (%res))IR";
  SubgraphRewriter fuse;
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  fuse.runOnGraph(graph);

  std::string pattern2 = R"IR(
    graph(%a, %b, %c, %d, %e, %f):
        %y0 : Tensor = aten::to(%a, %b, %c, %c)
        %y1 : Tensor = fb::batch_one_hot_lengths(%y0, %d, %e)
        %res : Tensor = aten::to(%y1, %f, %c, %c)
        return (%res))IR";
  std::string fused_pattern2 = R"IR(
    graph(%a, %b, %c, %d, %e, %f):
        %res : Tensor = fb::casted_batch_one_hot_lengths(%a, %d, %e)
        return (%res))IR";
  fuse.RegisterRewritePattern(pattern2, fused_pattern2);
  fuse.runOnGraph(graph);
}

void ConcatBatchMatMulBatchGather(std::shared_ptr<torch::jit::Graph>& graph) {
  // TODO:: check restrictions for inputs; outputs not used elsewhere
  std::string pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f):
        %y0 : Tensor = aten::stack(%a, %b)
        %y1 : Tensor = aten::transpose(%y0, %b, %c)
        %y2 : Tensor = aten::bmm(%y0, %y1)
        %y3 : Tensor = aten::flatten(%y2, %d, %e)
        %res : Tensor = aten::index_select(%y3, %b, %f)
        return (%res))IR";
  std::string fused_pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f):
        %res : Tensor = fb::concat_batch_matmul_batch_gather(%f, %a)
        return (%res))IR";
  SubgraphRewriter fuse;
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  fuse.runOnGraph(graph);
}

void ClipRangesGatherRangesLengthsToOffsets(
    std::shared_ptr<torch::jit::Graph>& graph) {
  // TODO:: check restrictions for inputs; outputs not used elsewhere
  std::string pattern = R"IR(
    graph(%a, %b, %c, %d):
        %y0 : Tensor = fb::clip_ranges(%b, %c)
        %y1 : Tensor, %y2 : Tensor = fb::gather_ranges(%a, %y0)
        %y3 : Tensor = fb::lengths_to_offsets(%y2, %d)
        return (%y3, %y1))IR";
  std::string fused_pattern = R"IR(
    graph(%a, %b, %c, %d):
        %y0 : Tensor, %y1 : Tensor = fb::clip_ranges_gather_lengths_to_offsets(%a, %b, %c, %d)
        return (%y1, %y0))IR";
  SubgraphRewriter fuse;
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  fuse.runOnGraph(graph);
}

void ClipRangesGather(std::shared_ptr<torch::jit::Graph>& graph) {
  // TODO:: check restrictions for inputs; outputs not used elsewhere
  // fuse without lengths-to-offsets
  std::string pattern = R"IR(
    graph(%a, %b, %c):
        %y0 : Tensor = fb::clip_ranges(%b, %c)
        %y1 : Tensor, %y2 : Tensor = fb::gather_ranges(%a, %y0)
        return (%y2, %y1))IR";
  std::string fused_pattern = R"IR(
    graph(%a, %b, %c):
        %y0 : Tensor, %y1 : Tensor = fb::clip_ranges_gather(%a, %b, %c)
        return (%y1, %y0))IR";
  SubgraphRewriter fuse;
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  fuse.runOnGraph(graph);
}

void ClipRangesGatherSigridHash(std::shared_ptr<torch::jit::Graph>& graph) {
  // TODO:: check restrictions for inputs; outputs not used elsewhere
  std::string pattern_1 = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g):
        %y0 : Tensor, %y1 : Tensor = fb::clip_ranges_gather_lengths_to_offsets(%a, %b, %c, %d)
        %y2 : Tensor = fb::sigrid_hash(%y0, %e, %f, %g)
        return (%y2, %y1))IR";
  std::string fused_pattern_1 = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g):
        %off : Tensor, %out : Tensor = fb::clip_ranges_gather_sigrid_hash_offsets(%b, %a, %c, %e, %f, %g, %d)
        return (%out, %off))IR";

  std::string pattern_2 = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g, %h):
        %y0 : Tensor, %y1 : Tensor = fb::clip_ranges_gather_lengths_to_offsets(%a, %b, %c, %d)
        %y2 : Tensor = fb::sigrid_hash_precompute(%y0, %e, %f, %g, %h)
        return (%y2, %y1))IR";
  std::string fused_pattern_2 = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g, %h):
        %off : Tensor, %out : Tensor = fb::clip_ranges_gather_sigrid_hash_precompute_offsets(%b, %a, %c, %e, %f, %g, %h, %d)
        return (%out, %off))IR";
  SubgraphRewriter fuse;
  fuse.RegisterRewritePattern(pattern_1, fused_pattern_1);
  fuse.runOnGraph(graph);

  fuse.RegisterRewritePattern(pattern_2, fused_pattern_2);
  fuse.runOnGraph(graph);
}

void ClipRangesGatherRangesSigridHash(
    std::shared_ptr<torch::jit::Graph>& graph) {
  std::string pattern_1 = R"IR(
    graph(%a, %b, %c, %d, %e, %f):
        %y0 : Tensor = fb::clip_ranges(%b, %c)
        %y1 : Tensor, %y2 : Tensor = fb::gather_ranges(%a, %y0)
        %y3 : Tensor = fb::sigrid_hash(%y1, %d, %e, %f)
        return (%y3, %y2))IR";
  std::string fused_pattern_1 = R"IR(
    graph(%a, %b, %c, %d, %e, %f):
        %off : Tensor, %out : Tensor = fb::clip_ranges_gather_sigrid_hash_v3(%b, %a, %c, %d, %e, %f)
        return (%out, %off))IR";

  std::string pattern_2 = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g):
        %y0 : Tensor = fb::clip_ranges(%b, %c)
        %y1 : Tensor, %y2 : Tensor = fb::gather_ranges(%a, %y0)
        %y3 : Tensor = fb::sigrid_hash_precompute(%y1, %d, %e, %f, %g)
        return (%y3, %y2))IR";
  std::string fused_pattern_2 = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g):
        %off : Tensor, %out : Tensor = fb::clip_ranges_gather_sigrid_hash_precompute_v3(%b, %a, %c, %d, %e, %f, %g)
        return (%out, %off))IR";

  SubgraphRewriter fuse;
  fuse.RegisterRewritePattern(pattern_1, fused_pattern_1);
  fuse.runOnGraph(graph);

  fuse.RegisterRewritePattern(pattern_2, fused_pattern_2);
  fuse.runOnGraph(graph);
}

void PrecomputeMultiplierShiftForSigridHash(
    std::shared_ptr<torch::jit::Graph>& graph) {
  std::string pattern = R"IR(
    graph(%a, %b, %c, %d):
        %y0 : Tensor = fb::sigrid_hash(%a, %b, %c, %d)
        return (%y0)
  )IR";
  std::string split_pattern = R"IR(
    graph(%a, %b, %c, %d):
        %y0 : Tensor = fb::sigrid_hash_compute_multipler_shift(%c)
        %y2 : Tensor = fb::sigrid_hash_precompute(%a, %b, %c, %y0, %d)
        return (%y2)
  )IR";
  SubgraphRewriter fuse;
  fuse.RegisterRewritePattern(pattern, split_pattern);
  fuse.runOnGraph(graph);
}

void ClipRangesGatherRangesX2SigridHash(
    std::shared_ptr<torch::jit::Graph>& graph) {
  // Placeholder is a dummy op used to capture the first subgraph
  std::string pattern = R"IR(
    graph(%ranges, %values, %max_length, %salt, %max_value, %hash_into_int32):
        %clipped : Tensor = fb::clip_ranges(%ranges, %max_length)
        %output : Tensor, %unused : Tensor = fb::gather_ranges(%values, %clipped)
        %sigrid_hash_out : Tensor = fb::sigrid_hash(%output, %salt, %max_value, %hash_into_int32)
        return (%sigrid_hash_out, %clipped))IR";
  std::string fused_pattern = R"IR(
    graph(%ranges, %values, %max_length, %salt, %max_value, %hash_into_int32):
        %sigrid_hash_out : Tensor, %clipped : Tensor = fb::placeholder(%ranges, %values, %max_length, %salt, %max_value, %hash_into_int32)
        return (%sigrid_hash_out, %clipped))IR";

  // the second gather_ranges can be eliminated because the `lengths` is
  // produces is identical to the lengths produced by
  // clip_ranges_gather_sigrid_hash_v3 (caveat, the fused ops makes some
  // simplifying assumptions about the ranges input)
  std::string pattern2 = R"IR(
    graph(%gather2_values, %ranges, %values, %max_length, %salt, %max_value, %hash_into_int32):
        %sigrid_hash_out : Tensor, %clipped : Tensor = fb::placeholder(%ranges, %values, %max_length, %salt, %max_value, %hash_into_int32)
        %unused : Tensor, %lengths : Tensor = fb::gather_ranges(%gather2_values, %clipped)
        return (%lengths, %sigrid_hash_out))IR";

  std::string fused_pattern2 = R"IR(
    graph(%gather2_values, %ranges, %values, %max_length, %salt, %max_value, %hash_into_int32):
        %lengths : Tensor, %sigrid_hash_out : Tensor = fb::clip_ranges_gather_sigrid_hash_v3(%ranges, %values, %max_length, %salt, %max_value, %hash_into_int32)
        return (%lengths, %sigrid_hash_out))IR";

  SubgraphRewriter fuse;
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  fuse.runOnGraph(graph);

  fuse.RegisterRewritePattern(pattern2, fused_pattern2);
  fuse.runOnGraph(graph);

  // reverse the ops that got fused in step 1 but not in step2
  fuse.RegisterRewritePattern(fused_pattern, pattern);
  fuse.runOnGraph(graph);
}

void ClipRangesGatherRangesX2SigridHashPrecompute(
    std::shared_ptr<torch::jit::Graph>& graph) {
  // Placeholder is a dummy op used to capture the first subgraph
  std::string pattern = R"IR(
    graph(%ranges, %values, %max_length, %salt, %max_value, %mul_shift, %hash_into_int32):
        %clipped : Tensor = fb::clip_ranges(%ranges, %max_length)
        %output : Tensor, %unused : Tensor = fb::gather_ranges(%values, %clipped)
        %sigrid_hash_out : Tensor = fb::sigrid_hash_precompute(%output, %salt, %max_value, %mul_shift, %hash_into_int32)
        return (%sigrid_hash_out, %clipped))IR";
  std::string fused_pattern = R"IR(
    graph(%ranges, %values, %max_length, %salt, %max_value, %mul_shift, %hash_into_int32):
        %sigrid_hash_out : Tensor, %clipped : Tensor = fb::placeholder(%ranges, %values, %max_length, %salt, %max_value, %mul_shift, %hash_into_int32)
        return (%sigrid_hash_out, %clipped))IR";

  // the second gather_ranges can be eliminated because the `lengths` is
  // produces is identical to the lengths produced by
  // clip_ranges_gather_sigrid_hash_v3 (caveat, the fused ops makes some
  // simplifying assumptions about the ranges input)
  std::string pattern2 = R"IR(
    graph(%gather2_values, %ranges, %values, %max_length, %salt, %max_value, %mul_shift, %hash_into_int32):
        %sigrid_hash_out : Tensor, %clipped : Tensor = fb::placeholder(%ranges, %values, %max_length, %salt, %max_value, %mul_shift, %hash_into_int32)
        %unused : Tensor, %lengths : Tensor = fb::gather_ranges(%gather2_values, %clipped)
        return (%lengths, %sigrid_hash_out))IR";

  std::string fused_pattern2 = R"IR(
    graph(%gather2_values, %ranges, %values, %max_length, %salt, %max_value, %mul_shift, %hash_into_int32):
        %lengths : Tensor, %sigrid_hash_out : Tensor = fb::clip_ranges_gather_sigrid_hash_precompute_v3(%ranges, %values, %max_length, %salt, %max_value, %mul_shift, %hash_into_int32)
        return (%lengths, %sigrid_hash_out))IR";

  SubgraphRewriter fuse;
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  fuse.runOnGraph(graph);

  fuse.RegisterRewritePattern(pattern2, fused_pattern2);
  fuse.runOnGraph(graph);

  // reverse the ops that got fused in step 1 but not in step2
  fuse.RegisterRewritePattern(fused_pattern, pattern);
  fuse.runOnGraph(graph);
}

void FuseInferenceOpsForSparseNN(std::shared_ptr<torch::jit::Graph>& graph) {
#ifdef FBCODE_CAFFE2
  SplitOutPrecomputeOpsForSparseNN(graph);

  ConcatAddMulReplaceNaNClip(graph);
  CastedBatchOneHotLengths(graph);
  ConcatBatchMatMulBatchGather(graph);

  ClipRangesGatherRangesLengthsToOffsets(graph);
  ClipRangesGatherSigridHash(graph);
  ClipRangesGatherRangesSigridHash(graph);

  ClipRangesGatherRangesX2SigridHash(graph);
  ClipRangesGatherRangesX2SigridHashPrecompute(graph);

  // prioritize clip_ranges+gather_ranges+sigrid_hash fusion over
  // clip_ranges+gather_ranges
  ClipRangesGather(graph);
#endif
}

void SplitOutPrecomputeOpsForSparseNN(
    std::shared_ptr<torch::jit::Graph>& graph) {
#ifdef FBCODE_CAFFE2
  PrecomputeMultiplierShiftForSigridHash(graph);
  ConstantPropagation(graph);
  ConstantPooling(graph);
#endif
}

TORCH_LIBRARY_FRAGMENT(static_runtime, m) {
  m.def("static_runtime::pure_inputs() -> Tensor", []() -> at::Tensor {
    return at::randn({1});
  });
  m.def("static_runtime::permute_copy(Tensor self, int[] dims) -> Tensor");
  m.def(
      "static_runtime::to_copy(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor");
}

void ReplaceWithCopy(std::shared_ptr<torch::jit::Graph>& graph) {
  auto* fake_input =
      graph->insert(Symbol::fromQualString("static_runtime::pure_inputs"), {});
  fake_input->node()->moveBefore(*graph->nodes().begin());

  std::vector<std::pair<Value*, Use>> old_inputs;
  for (auto* input : graph->inputs()) {
    for (const auto& use : input->uses()) {
      old_inputs.emplace_back(std::make_pair(input, use));
    }
    input->replaceAllUsesWith(fake_input);
  }

  AliasDb db(graph);
  for (const auto& p : old_inputs) {
    p.second.user->replaceInput(p.second.offset, p.first);
  }
  fake_input->node()->destroy();

  const std::map<c10::Symbol, c10::Symbol> supported = {
      {c10::Symbol::fromQualString("aten::permute"),
       c10::Symbol::fromQualString("static_runtime::permute_copy")},
      {c10::Symbol::fromQualString("aten::narrow"),
       c10::Symbol::fromQualString("aten::narrow_copy")},
      {c10::Symbol::fromQualString("aten::to"),
       c10::Symbol::fromQualString("static_runtime::to_copy")}};
  std::vector<std::pair<Node*, Node*>> replacement;
  for (auto* n : graph->nodes()) {
    if (!supported.count(n->kind())) {
      continue;
    }
    DCHECK(n->outputs().size() == 1);

    // In cases of having in-place ops in the graph, only replace the op with
    // the copy version for ops with input with number of use == 1. Example:
    //
    // def forward(self, inp: Tensor, shape: List[int]):
    //   a = inp + inp
    //   b = a.reshape(shape)
    //   c = b.sigmoid_()
    //   d = c + c
    //   e = a + a
    //   f = b + b
    //   return (d, e, f)
    //
    // b and c are aliases of a, sigmoid_ changes b, c, as well as a. e should
    // equal to d in this case. If we replace reshape with the copy version, b
    // and c are no longer aliases of a, the value of e would change as a
    // result. To keep static runtime consistent with the jit interpreter, here
    // we choose not to replace reshape with the copy version
    auto* in = n->input(0);
    if (in->uses().size() > 1) {
      continue;
    }

    auto* out = n->output();
    if (db.mayContainAlias({out}, graph->outputs())) {
      continue;
    }
    auto new_symbol = supported.at(n->kind());
    auto* new_node = graph->create(new_symbol, n->outputs().size());
    new_node->insertBefore(n);
    for (auto* input : n->inputs()) {
      new_node->addInput(input);
    }
    replacement.emplace_back(std::make_pair(n, new_node));
  }

  for (const auto& p : replacement) {
    auto* old_node = p.first;
    auto* new_node = p.second;
    old_node->replaceAllUsesWith(new_node);
    old_node->destroy();
  }
}

void FuseSigridTransformsListUnpack(std::shared_ptr<torch::jit::Graph>& graph) {
  auto nodes = graph->nodes();
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    Node* sigrid_node = *it;
    auto kind = sigrid_node->kind();
    // TODO: make it work the TorchBind version
    if (strcmp(kind.toQualString(), "fb::sigrid_transforms") == 0) {
      const Value* sigrid_out = sigrid_node->outputs()[0];
      if (sigrid_out->uses().size() > 1) {
        continue;
      }

      Node* list_unpack_node = sigrid_out->uses()[0].user;
      if (list_unpack_node->kind() != prim::ListUnpack) {
        continue;
      }

      auto list_unpack_outputs = list_unpack_node->outputs();
      if (list_unpack_outputs.empty()) {
        continue;
      }

      // handle outputs
      for (Value* out : list_unpack_outputs) {
        Value* new_out = sigrid_node->addOutput();
        new_out->copyMetadata(out);
        out->replaceAllUsesWith(new_out);
      }

      auto it_next = it;
      ++it_next; // it_next points to list_unpack
      it_next.destroyCurrent(); // remove list_unpack

      sigrid_node->eraseOutput(0);
    }
  }
}

} // namespace jit
} // namespace torch
