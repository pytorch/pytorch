#include <torch/nativert/executor/memory/AliasAnalyzer.h>

#include <c10/util/Enumerate.h>

namespace torch::nativert {

AliasAnalyzer::AliasAnalyzer(
    const Graph& graph,
    const c10::FastMap<std::string /* target */, FunctionSchema>& schemas) {
  for (const auto&& [i, node] : c10::enumerate(graph.nodes())) {
    for (const auto& input : node.inputs()) {
      create_or_update_lifetime(input.value, i);
    }

    for (const auto& output : node.outputs()) {
      create_or_update_lifetime(output, i);
    }

    if (update_aliases_if_packed_listunpack(node, i) /* applied? */) {
      continue;
    }

    maybe_update_aliases_from_schema(node, schemas);
  }

  maybe_extend_lifetimes(graph);

  // squash_deep_aliases this will populate aliases_
  // with a mapping from each alias to its backed
  // source (i.e., the value that owns the underlying
  // dataptr for said alias)
  squash_deep_aliases(graph);

  // set all non-aliasing outputs. outputs
  // that are aliased will be set later when
  // lifetimes are extended
  for (const auto* output : graph.outputs()) {
    if (!is_alias(output)) {
      values_associated_with_outputs_.emplace(output);
    }
  }

  log_state();

  alive_values_at_time_.resize(graph.nodes().size());
  for (const auto& [v, lifetime] : lifetimes_) {
    for (const auto t : c10::irange(lifetime.start, lifetime.end + 1)) {
      alive_values_at_time_[t].emplace_back(v);
    }
  }
} // namespace torch::nativert

bool /* applied */ AliasAnalyzer::update_aliases_if_packed_listunpack(
    const Node& node,
    size_t i) {
  if (node.target() != "prim.ListUnpack") {
    return false;
  }

  const auto* list = node.inputs()[0].value;

  // we can't infer about how this list was made in this case
  // so fallback to default always-aliasing behaviour
  if (const auto* p = list->producer(); p && p->target() != "prim.ListPack") {
    return false;
  }

  const auto& list_elems = list->getListElements();
  TORCH_CHECK(list_elems.size() == node.numOutputs());

  for (const auto j : c10::irange(node.numOutputs())) {
    const Value* input = list_elems.at(j);
    const Value* output = node.outputs().at(j);

    TORCH_CHECK(input != output);

    create_or_update_lifetime(input, i);
    create_or_update_lifetime(output, i);

    aliases_[output].emplace(input);
  }

  return true;
}

void AliasAnalyzer::maybe_update_aliases_from_schema(
    const Node& node,
    const c10::FastMap<std::string /* target */, FunctionSchema>& schemas) {
  std::function<bool(size_t, size_t)> is_alias =
      []([[maybe_unused]] size_t input_idx,
         [[maybe_unused]] size_t output_idx) { return true; };

  const FunctionSchema* schema = nullptr;
  if (auto schemaIt = schemas.find(std::string(node.target()));
      schemaIt != schemas.end()) {
    schema = &schemaIt->second;
  }

  if (!schema) {
    VLOG(1) << "schema not found for " << node.target()
            << " assuming worst case aliasing";
  }

  for (size_t j = 0; j < node.numInputs(); j += 1) {
    for (size_t k = 0; k < node.numOutputs(); k += 1) {
      const Value* input = node.inputs().at(j).value;
      const Value* output = node.outputs().at(k);

      if (!schema || schema->alias(j, k)) {
        VLOG(1) << node.target()
                << " may contain input/output alias: " << input->id() << " -> "
                << output->id();
        aliases_[output].emplace(input);
      }
    }
  }
}

void AliasAnalyzer::create_or_update_lifetime(const Value* value, size_t i) {
  if (auto [lifetimeIt, inserted] = lifetimes_.try_emplace(value, i, i);
      !inserted) {
    lifetimeIt->second.end = i;
  }
}

void AliasAnalyzer::squash_deep_aliases(const Graph& graph) {
  for (auto& node : graph.nodes()) {
    for (const auto& output : node.outputs()) {
      auto aliasIt = aliases_.find(output);
      if (aliasIt == aliases_.end()) {
        continue;
      }

      c10::FastSet<const Value*> filtered_srcs;

      auto& srcs = aliasIt->second;
      for (const auto* src : srcs) {
        // check if this source is an alias itself,
        // making 'output' a deep alias (i.e.,
        // an alias of an alias)

        // we want aliases_[x] to return the value from which x
        // inherits its dataptr.
        // as such, we want to add values that do not meet this
        // criteria (i.e., those that are aliases).
        // in practice, there can only be 1 value that meets this
        // criteria (at a time), but there are some cases where
        // this is ambiguous (e.g., where the spec doesn't exist,
        // dealing with variadics)
        auto srcAliasIt = aliases_.find(src);
        if (srcAliasIt == aliases_.end()) {
          filtered_srcs.emplace(src);
          continue;
        }

        // since we are going from the beginning of the graph
        // to the end of the graph we can assume that these
        // aliases, which have already been visited, have already
        // been squashed.
        auto& srcs_of_src = srcAliasIt->second;
        for (const auto* src_of_src : srcs_of_src) {
          // if the source of the source is not an alias
          // (i.e., it has ownership over it's data ptr)
          // then we want to add it as a source of 'output'
          if (aliases_.find(src_of_src) == aliases_.end()) {
            filtered_srcs.emplace(src_of_src);
          }
        }
      }

      srcs = std::move(filtered_srcs);
    }
  }
}

void AliasAnalyzer::maybe_extend_lifetimes(const Graph& graph) {
  c10::FastSet<const Value*> extended;

  for (auto nodeIt = graph.nodes().rbegin(); nodeIt != graph.nodes().rend();
       ++nodeIt) {
    const auto& inputs = nodeIt->inputs();
    for (const auto& input : inputs) {
      if (auto aliasIt = aliases_.find(input.value);
          aliasIt != aliases_.end()) {
        const auto& alias = aliasIt->second;
        for (const auto& src : alias) {
          if (extended.find(src) != extended.end()) {
            continue;
          }

          auto& eol = lifetimes_[src].end;
          eol = lifetimes_[input.value].end;

          VLOG(1) << "extended EOL of value " << src->id() << " to " << eol;

          extended.emplace(src);

          if (aliases_.find(src) == aliases_.end() &&
              eol == graph.nodes().size() - 1 /* aliases output */) {
            values_associated_with_outputs_.emplace(src);
          }
        }
      }
    }
  }
}

void AliasAnalyzer::log_state() const {
  if (!VLOG_IS_ON(
          1) /* this is usually too large to be logged with VLOG directly */) {
    return;
  }

  std::cout << [&]() -> std::string {
    std::ostringstream ss;
    ss << "[nativert layout planner] AliasAnalyzer ran....\n";
    ss << "lifetimes:\n";

    for (const auto& [v, lifetime] : lifetimes_) {
      ss << "  " << v->name() << ": [" << lifetime.start << ", " << lifetime.end
         << "]\n";
    }

    ss << "\naliases:\n";
    for (const auto& [v, alias] : aliases_) {
      ss << "  " << v->name() << " -> ";
      for (const auto* a : alias) {
        ss << a->name() << ", ";
      }
      ss << '\n';
    }

    ss << '\n';

    return ss.str();
  }() << std::flush;
}

} // namespace torch::nativert
