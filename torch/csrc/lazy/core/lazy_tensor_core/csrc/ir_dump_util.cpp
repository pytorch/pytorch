#include "lazy_tensor_core/csrc/ir_dump_util.h"

#include <c10/util/Optional.h>

#include <regex>
#include <sstream>
#include <unordered_map>

#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"
#include "lazy_tensor_core/csrc/ir_util.h"
#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/computation_client/debug_macros.h"

namespace torch_lazy_tensors {
namespace ir {
namespace {

using NodeIdMap = std::unordered_map<const Node*, size_t>;

struct AttrTag {
  std::string name;
  std::string value;
  std::string::size_type pos;
};

std::string::size_type SkipTagSeparator(const std::string& node_string,
                                        std::string::size_type pos) {
  return node_string.compare(pos, 2, ", ") == 0 ? pos + 2 : pos;
}

c10::optional<AttrTag> ParseAttrTag(const std::string& node_string,
                                    std::string::size_type pos) {
  const std::regex tag_regex("^([a-zA-Z0-9_]+)=");
  std::smatch match;
  if (!std::regex_search(node_string.begin() + pos, node_string.end(), match,
                         tag_regex)) {
    return c10::nullopt;
  }

  std::string::size_type vpos = match[1].second - node_string.begin() + 1;
  int nested_open = -1;
  int nested_close = -1;
  size_t nest_count = 1;
  AttrTag tag;
  tag.name = match[1].str();
  for (pos = vpos; pos < node_string.size(); ++pos) {
    if (nested_open < 0) {
      if (SkipTagSeparator(node_string, pos) != pos) {
        break;
      }
      switch (node_string[pos]) {
        case '(':
          nested_open = node_string[pos];
          nested_close = ')';
          break;
        case '[':
          nested_open = node_string[pos];
          nested_close = ']';
          break;
        case '{':
          nested_open = node_string[pos];
          nested_close = '}';
          break;
      }
    } else if (node_string[pos] == nested_close) {
      --nest_count;
      if (nest_count == 0) {
        nest_count = 1;
        nested_open = nested_close = -1;
      }
    } else if (node_string[pos] == nested_open) {
      ++nest_count;
    }
  }
  tag.value = node_string.substr(vpos, pos - vpos);
  tag.pos = pos;
  return tag;
}

NodeIdMap GenerateIdMap(lazy_tensors::Span<const Node* const> post_order) {
  NodeIdMap id_map;
  for (auto node : post_order) {
    LTC_CHECK(id_map.emplace(node, id_map.size()).second) << node->ToString();
  }
  return id_map;
}

std::unordered_map<const Node*, size_t> GetRootsIds(
    lazy_tensors::Span<const Node* const> roots) {
  std::unordered_map<const Node*, size_t> roots_ids;
  for (size_t i = 0; i < roots.size(); ++i) {
    roots_ids[roots[i]] = i;
  }
  return roots_ids;
}

c10::optional<size_t> GetRootNodeId(
    const Node* node,
    const std::unordered_map<const Node*, size_t>& roots_ids) {
  auto it = roots_ids.find(node);
  if (it == roots_ids.end()) {
    return c10::nullopt;
  }
  return it->second;
}

std::vector<AttrTag> GetNodeTags(const Node* node) {
  std::string node_string = node->ToString();
  std::string op_string = node->op().ToString();
  std::string::size_type pos = node_string.find(op_string);
  LTC_CHECK_NE(pos, std::string::npos) << node_string << " : " << op_string;
  pos += op_string.size();
  std::vector<AttrTag> tags;
  for (;;) {
    pos = SkipTagSeparator(node_string, pos);
    auto tag = ParseAttrTag(node_string, pos);
    if (!tag) {
      break;
    }
    pos = tag->pos;
    tags.push_back(std::move(*tag));
  }
  return tags;
}

std::string GenerateDotNodeLabel(
    const Node* node,
    const std::unordered_map<const Node*, size_t>& roots_ids) {
  static const size_t kMaxValueSize = 64;
  std::stringstream ss;
  ss << node->op() << "\\n" << node->shape();
  for (auto& tag : GetNodeTags(node)) {
    ss << "\\n" << tag.name << "=";
    if (tag.value.size() < kMaxValueSize) {
      ss << tag.value;
    } else {
      ss << tag.value.substr(0, kMaxValueSize) << "...";
    }
  }
  auto opt_root_id = GetRootNodeId(node, roots_ids);
  if (opt_root_id) {
    ss << "\\nROOT=" << *opt_root_id;
  }
  return ss.str();
}

std::string GenerateDotNodeSpec(
    const Node* node,
    const std::unordered_map<const Node*, size_t>& roots_ids) {
  std::stringstream ss;
  ss << "label=\"" << GenerateDotNodeLabel(node, roots_ids) << "\"";
  return ss.str();
}

std::string GenerateTextNodeSpec(const Node* node, const NodeIdMap& id_map) {
  std::stringstream ss;
  ss << node->shape() << " " << node->op() << "(";
  size_t count = 0;
  for (auto& output : node->operands()) {
    if (count > 0) {
      ss << ", ";
    }
    ss << "%" << id_map.at(output.node);
    if (output.node->num_outputs() > 1) {
      ss << "." << output.index;
    }
    ++count;
  }
  ss << ")";
  for (auto& tag : GetNodeTags(node)) {
    ss << ", " << tag.name << "=" << tag.value;
  }
  return ss.str();
}

}  // namespace

std::string DumpUtil::ToDot(lazy_tensors::Span<const Node* const> nodes) {
  auto post_order = Util::ComputePostOrder(nodes);
  return PostOrderToDot(post_order, nodes);
}

std::string DumpUtil::PostOrderToDot(
    lazy_tensors::Span<const Node* const> post_order,
    lazy_tensors::Span<const Node* const> roots) {
  std::unordered_map<const Node*, size_t> roots_ids = GetRootsIds(roots);
  NodeIdMap id_map = GenerateIdMap(post_order);
  std::stringstream ss;
  ss << "digraph G {\n";
  for (auto node : post_order) {
    ss << "  node" << id_map.at(node) << " ["
       << GenerateDotNodeSpec(node, roots_ids) << "]\n";
  }
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
    const Node* node = *it;
    size_t id = id_map.at(node);
    for (size_t i = 0; i < node->operands().size(); ++i) {
      const ir::Output& output = node->operand(i);
      ss << "  node" << id_map.at(output.node) << " -> node" << id;
      if (node->operands().size() > 1) {
        ss << " [label=\"i=" << i;
        if (output.node->num_outputs() > 1) {
          ss << ",o=" << output.index;
        }
        ss << "\"]\n";
      } else {
        if (output.node->num_outputs() > 1) {
          ss << " [label=\"o=" << output.index << "\"]";
        }
        ss << "\n";
      }
    }
  }
  ss << "}\n";
  return ss.str();
}

std::string DumpUtil::ToText(lazy_tensors::Span<const Node* const> nodes) {
  auto post_order = Util::ComputePostOrder(nodes);
  return PostOrderToText(post_order, nodes);
}

std::string DumpUtil::PostOrderToText(
    lazy_tensors::Span<const Node* const> post_order,
    lazy_tensors::Span<const Node* const> roots) {
  std::unordered_map<const Node*, size_t> roots_ids = GetRootsIds(roots);
  NodeIdMap id_map = GenerateIdMap(post_order);
  std::stringstream ss;
  ss << "IR {\n";
  for (auto node : post_order) {
    auto opt_root_id = GetRootNodeId(node, roots_ids);
    ss << "  %" << id_map.at(node) << " = "
       << GenerateTextNodeSpec(node, id_map);
    if (opt_root_id) {
      ss << ", ROOT=" << *opt_root_id;
    }
    ss << "\n";
  }
  ss << "}\n";
  return ss.str();
}

std::string DumpUtil::ToBackend(lazy_tensors::Span<const Value> values,
                                const Device& device) {
  auto lowering_ctx = ir::LoweringContext::Create("IrToBackend", device);
  for (auto& ir_value : values) {
    lowering_ctx->AddResult(ir_value);
  }
  auto computation = ConsumeValue(lowering_ctx->Build());
  return ConsumeValue(
      compiler::getBackendRegistrar()->GetComputationBackendText(
          computation.get()));
}

}  // namespace ir
}  // namespace torch_lazy_tensors
