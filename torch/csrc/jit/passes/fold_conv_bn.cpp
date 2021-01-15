#include <torch/csrc/jit/passes/fold_conv_bn.h>

#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/quantization/helper.h>

#include <stack>

namespace torch {
namespace jit {

std::tuple<at::Tensor, at::Tensor> computeUpdatedConvWeightAndBias(
    const ConvBNParameters& p) {
  at::Tensor bn_var_rsqrt = at::rsqrt(p.bn_rv + p.bn_eps);
  const int64_t ndim = p.conv_w.dim();
  at::DimVector sizes(ndim, 1);
  sizes.at(0) = -1;
  at::Tensor new_w = p.conv_w * (p.bn_w * bn_var_rsqrt).reshape(sizes);
  at::Tensor new_b = (p.conv_b - p.bn_rm) * bn_var_rsqrt * p.bn_w + p.bn_b;
  return std::make_tuple(new_w, new_b);
}

namespace {
using graph_rewrite_helper::PatternInfo;

static bool hastensor(Module& m, const char* name) {
  return m.hasattr(name) && m.attr(name).isTensor();
}

void replaceConvBiasWithGetAttr(Module& module) {
  auto graph = module.get_method("forward").graph();
  // Only looks fors _convolution pattern.
  // Thus assumes that tracing will have always gotten rid of aten::conv2d or
  // aten::conv3d. If it did not, BN folding will fail.
  const PatternInfo& pattern_convolution = PatternInfo::parse_from_str(R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %conv_out = aten::_convolution(%a, %w, %b, %stride, %padding, %dilation,
            %transposed, %output_padding, %groups, %benchmark, %deterministic, %cudnn_enabled, %allow_tf32)
        return (%conv_out) )");
  const PatternInfo& pattern_convolution_deprecated =
      PatternInfo::parse_from_str(R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %conv_out = aten::_convolution(%a, %w, %b, %stride, %padding, %dilation,
            %transposed, %output_padding, %groups, %benchmark, %deterministic, %cudnn_enabled)
        return (%conv_out) )");
  auto replace_pattern = [&](const PatternInfo& pattern_convolution) {
    const Graph& pattern_convolution_graph = *pattern_convolution.pattern_graph;
    const auto& convolution_vmap = pattern_convolution.vmap;

    const auto& matches = findPatternMatches(pattern_convolution_graph, *graph);
    for (const auto& match : matches) {
      // We come here only if the bias was not present in the module.
      // In that case, the corresponding graph will not have getAttr("bias")
      // Insert that in the graph.
      // And change _convolution to take the new value.
      auto conv_node =
          match.values_map.at(convolution_vmap.at("conv_out"))->node();
      WithInsertPoint ins(conv_node);
      Value* bias_attr_val = graph->insertGetAttr(graph->inputs()[0], "bias")
                                 ->setType(TensorType::get());
      constexpr size_t conv_bias_index = 2;
      conv_node->replaceInput(conv_bias_index, bias_attr_val);
    }
  };
  replace_pattern(pattern_convolution);
  replace_pattern(pattern_convolution_deprecated);
}

void addBiasForConvIfNone(Module& module, const std::string& pattern_name) {
  auto t = module.type()->expect<ClassType>();

  const std::string real_typename = t->name()->qualifiedName();
  const std::string demangled_typename = removeTorchMangle(real_typename);
  bool is_floating_point_conv =
      ((demangled_typename == "__torch__.torch.nn.modules.conv.Conv1d") ||
       (demangled_typename == "__torch__.torch.nn.modules.conv.Conv2d") ||
       (demangled_typename == "__torch__.torch.nn.modules.conv.Conv3d"));

  if (is_floating_point_conv) {
    if (!t->hasAttribute("bias")) {
      auto optional_tensor_type = OptionalType::create(TensorType::get());
      t->addAttribute("bias", optional_tensor_type, true);
      auto optional_tensor = c10::optional<at::Tensor>();
      module.setattr("bias", optional_tensor);
      replaceConvBiasWithGetAttr(module);
    }
  }
  for (Module m : module.children()) {
    addBiasForConvIfNone(m, pattern_name);
  }
}

class FoldConvBatchNormHelper {
 public:
  /**
   * In this step we find all Conv - BatchNorm patterns in the graph
   * and extract the corresponding parameters for these two modules,
   * and record informations for the modifications of the graph without
   * actually performing these modifications.
   */
  void analyze(Module& module, const PatternInfo& pattern);
  /**
   * In this step we perform all the modifications including
   * setting the attributes for conv module, rewriting values
   * and deleting nodes in the graph
   */
  void transform();

 private:
  bool tryExtractingConvBNParameters(
      Module& conv,
      Module& bn,
      ConvBNParameters& r);

  std::unordered_map<ModulePtr, std::tuple<at::Tensor, at::Tensor>>
      conv_module_and_params_;

  // A map from graph to a list of tuple of paths of matched conv and bn module
  // e.g. if we have a graph `g` containing following code
  // x = self.sub.conv1(..)
  // x = self.sub.bn1(..)
  // x = self.sub.conv2(..)
  // x = self.sub.bn2(..)
  // then the value for graph `g` in this map will be:
  // [(['sub', 'conv1'], ['sub', 'bn1']), (['sub', 'conv2'], ['sub', 'bn2'])]
  // the first entry of the list is the paths to first conv-bn match
  // the second entry of the list is the path to second match
  std::unordered_map<
      Graph*,
      std::vector<
          std::tuple<std::vector<std::string>, std::vector<std::string>>>>
      conv_bn_paths_;

  std::unordered_map<Value*, Value*> rewrite_map_;
  std::vector<Value*> values_to_rewrite_;
  std::unordered_set<Node*> nodes_to_delete_;
};

bool extractOptionalBNParams(const script::Module& bn, ConvBNParameters& r) {
  auto bn_forward = bn.get_method("forward");
  auto graph = bn_forward.graph();
  const PatternInfo& pattern_bn = PatternInfo::parse_from_str(R"(
      graph(%a, %weight, %bias, %running_mean, %running_var,
          %training, %momentum, %eps, %cudnn_enabled):
        %bn_out = aten::batch_norm(%a, %weight, %bias, %running_mean,
            %running_var, %training, %momentum, %eps, %cudnn_enabled)
        return (%bn_out) )");
  const Graph& pattern_bn_graph = *pattern_bn.pattern_graph;
  const auto& bn_vmap = pattern_bn.vmap;

  const auto& matches = findPatternMatches(pattern_bn_graph, *graph);

  if (matches.size() > 1) {
    return false;
  }

  if (bn.hasattr("eps")) {
    r.bn_eps = bn.attr("eps").toDouble();
  } else {
    auto optional_eps = toIValue(matches[0].values_map.at(bn_vmap.at("eps")));
    if (!optional_eps) {
      return false;
    }
    r.bn_eps = optional_eps.value().toDouble();
  }
  r.bn_w = at::ones_like(bn.attr("running_mean").toTensor());
  if (bn.hasattr("weight")) {
    if (bn.attr("weight").isTensor()) {
      r.bn_w = bn.attr("weight").toTensor();
    }
  } else {
    auto optional_bn_weight =
        toIValue(matches[0].values_map.at(bn_vmap.at("weight")));
    if (!optional_bn_weight) {
      return false;
    }
    if (optional_bn_weight.value().isTensor()) {
      r.bn_w = optional_bn_weight.value().toTensor();
    }
  }
  r.bn_b = at::zeros_like(bn.attr("running_mean").toTensor());
  if (bn.hasattr("bias")) {
    if (bn.attr("bias").isTensor()) {
      r.bn_b = bn.attr("bias").toTensor();
    }
  } else {
    auto optional_bn_bias =
        toIValue(matches[0].values_map.at(bn_vmap.at("bias")));
    if (!optional_bn_bias) {
      return false;
    }

    if (optional_bn_bias.value().isTensor()) {
      r.bn_b = optional_bn_bias.value().toTensor();
    }
  }
  return true;
}

bool FoldConvBatchNormHelper::tryExtractingConvBNParameters(
    Module& conv,
    Module& bn,
    ConvBNParameters& r) {
  if (!hastensor(conv, "weight") || !conv.hasattr("bias") ||
      !hastensor(bn, "running_mean") || !hastensor(bn, "running_var")) {
    return false;
  }

  r.bn_rm = bn.attr("running_mean").toTensor();
  r.bn_rv = bn.attr("running_var").toTensor();
  if (!extractOptionalBNParams(bn, r)) {
    return false;
  }

  r.conv_w = conv.attr("weight").toTensor();
  r.conv_b = at::zeros_like(r.bn_rm);
  auto bias_opt = conv.attr("bias").toOptional<at::Tensor>();
  if (bias_opt) {
    r.conv_b = *bias_opt;
  }

  return true;
}

void FoldConvBatchNormHelper::analyze(
    Module& module,
    const PatternInfo& pattern) {
  const Graph& pattern_graph = *pattern.pattern_graph;
  const auto& vmap = pattern.vmap;
  Value* pattern_conv_out = vmap.at("conv_out");
  Value* pattern_bn_out = vmap.at("bn_out");
  Value* pattern_bn_submodule = vmap.at("batchnorm");
  Node* pattern_conv = pattern_conv_out->node();
  Node* pattern_bn = pattern_bn_out->node();

  // We will put submodules into this worklist and keep processing items from it
  // one by one. We start by just putting the top module there.
  std::stack<Module> worklist({module});
  while (!worklist.empty()) {
    Module current = worklist.top();
    worklist.pop();

    // Queue submodules for processing
    for (const Module& submodule : current.children()) {
      worklist.push(submodule);
    }

    // Process all method of the current module
    for (auto& method : current.get_methods()) {
      GRAPH_DUMP(
          current.type()->name()->name() + "::" + method.name() +
              "() before Conv-BatchNorm folding",
          method.graph());
      const auto& matches = findPatternMatches(pattern_graph, *method.graph());

      GRAPH_DEBUG("number of Conv-BatchNorm matches: ", matches.size());
      Graph* g = method.graph().get();
      if (!conv_bn_paths_.count(g)) {
        // This is to make sure we don't visit one graph multiple times
        conv_bn_paths_[g] = {};
        for (const Match& match : matches) {
          if (!std::all_of(
                  pattern.filters.begin(),
                  pattern.filters.end(),
                  [&](const MatchFilter& f) { return f(match, vmap); })) {
            continue;
          }
          GRAPH_DEBUG("Checking next match...");
          // Get the conv and bn submodule
          Node* matched_conv = match.nodes_map.at(pattern_conv);
          Node* matched_bn = match.nodes_map.at(pattern_bn);
          Node* matched_bn_submodule =
              match.values_map.at(pattern_bn_submodule)->node();
          Value* conv_instance = matched_conv->input(0);
          Value* bn_instance = matched_bn->input(0);
          Value* self = g->inputs()[0];
          auto conv_module_path = getModuleAccessPath(conv_instance, self);
          auto bn_module_path = getModuleAccessPath(bn_instance, self);
          Module conv_submodule = findChildModule(current, conv_module_path);
          Module bn_submodule = findChildModule(current, bn_module_path);

          ConvBNParameters params;
          if (!tryExtractingConvBNParameters(
                  conv_submodule, bn_submodule, params)) {
            GRAPH_DEBUG(
                "Conv and BN modules didn't have all required parameters or attributes...");
            continue;
          }
          conv_bn_paths_[g].push_back(
              std::make_tuple(conv_module_path, bn_module_path));
          // We are using a separate vector for saving Values we want to rewrite
          // to make sure that the order in which we perform these
          // transformations is deterministic. Iterating through keys of
          // rewrite_map would result in non-determinism that might not manifest
          // as a bug now, but can bite us later.
          values_to_rewrite_.push_back(matched_bn->output());
          rewrite_map_[matched_bn->output()] = matched_conv->output();
          GRAPH_UPDATE(
              "Rewriting %",
              matched_bn->output()->debugName(),
              " with %",
              matched_conv->output()->debugName());

          nodes_to_delete_.insert(matched_bn);
          nodes_to_delete_.insert(matched_bn_submodule);
          GRAPH_UPDATE("Deleting ", *matched_bn);
          GRAPH_UPDATE("Deleting ", *matched_bn_submodule);

          auto slot = conv_submodule.type()->getAttributeSlot("bias");
          TORCH_CHECK(
              conv_submodule.type()->is_parameter(slot),
              "Expected conv module to have a bias parameter");
        } // matches
      }

      for (const auto& conv_bn : conv_bn_paths_.at(g)) {
        Module conv_submodule = findChildModule(current, std::get<0>(conv_bn));
        Module bn_submodule = findChildModule(current, std::get<1>(conv_bn));

        ConvBNParameters params;
        TORCH_INTERNAL_ASSERT(tryExtractingConvBNParameters(
            conv_submodule, bn_submodule, params));
        auto new_w_b = computeUpdatedConvWeightAndBias(params);
        conv_module_and_params_[conv_submodule._ivalue()] = new_w_b;
      } // conv_bn module
    } // methods
  } // while
}

void FoldConvBatchNormHelper::transform() {
  for (const auto& item : conv_module_and_params_) {
    Module conv(item.first);
    auto w_b = item.second;
    conv.setattr("weight", std::get<0>(w_b));
    conv.setattr("bias", std::get<1>(w_b));
  }

  // Perform planned rewritings
  for (auto v : values_to_rewrite_) {
    v->replaceAllUsesWith(rewrite_map_.at(v));
  }

  // Perform planned deletions
  for (auto n : nodes_to_delete_) {
    n->removeAllInputs();
  }
  for (auto n : nodes_to_delete_) {
    n->destroy();
  }
}

} // namespace

Module FoldConvBatchNorm(const Module& module) {
  Module m = module.clone();

  addBiasForConvIfNone(m, "Conv2d");
  addBiasForConvIfNone(m, "Conv3d");
  // Conv2d + BatchNorm2d
  const PatternInfo pattern2d = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %conv, %batchnorm):
    %conv_out = prim::CallMethod[name="forward"](%conv, %input)
    %bn_out = prim::CallMethod[name="forward"](%batchnorm, %conv_out)
    return (%bn_out))",
      {is_conv2d_module, is_batchnorm2d_module});
  // Conv3d + BatchNorm3d
  const PatternInfo pattern3d = PatternInfo::parse_from_str(
      R"(
graph(%self, %input, %conv, %batchnorm):
    %conv_out = prim::CallMethod[name="forward"](%conv, %input)
    %bn_out = prim::CallMethod[name="forward"](%batchnorm, %conv_out)
    return (%bn_out))",
      {is_conv3d_module, is_batchnorm3d_module});

  const std::vector<std::reference_wrapper<const PatternInfo>> patterns = {
      pattern2d, pattern3d};
  for (const auto& pattern : patterns) {
    FoldConvBatchNormHelper h;
    h.analyze(m, pattern);
    h.transform();
  }
  return m;
}

} // namespace jit
} // namespace torch
