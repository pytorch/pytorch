#include "torch/csrc/nativert/common/Pytree.h"
#include "torch/csrc/nativert/common/RecordFunction.h"

#include <iterator>
#include <string_view>

#include <ATen/core/ivalue.h>
#include <c10/util/Synchronized.h>
#include <nlohmann/json.hpp> // @manual=fbsource//third-party/nlohmann-json:nlohmann-json

namespace torch::nativert {

namespace {
inline constexpr int kDefaultTreeSpecSerializationProtocol = 1;

c10::IValue dynamicToIValue(const nlohmann::json& obj) {
  if (obj.is_string()) {
    return obj.get<std::string>();
  } else if (obj.is_number_integer()) {
    return obj.get<int64_t>();
  } else {
    TORCH_CHECK(false, "Unsupported dynamic type: ", obj);
  }
}

void treeFlatten(
    const c10::IValue& tree,
    const TreeSpec& spec,
    std::vector<c10::IValue>& leaves) {
  if (spec.isLeaf()) {
    leaves.push_back(tree);
    return;
  }
  auto flattenFn = spec.nodeDefCache().flattenFn;
  flattenFn(tree, spec, leaves);
}

class PytreeNodeRegistry {
 public:
  PytreeNodeRegistry() {
    // Add some law of physics here.
    registerNode(
        "builtins.tuple",
        NodeDef{
            [](const c10::IValue& tree,
               const TreeSpec& spec,
               std::vector<c10::IValue>& leaves) {
              const auto& tuple = tree.toTupleRef().elements();
              TORCH_CHECK_EQ(tuple.size(), spec.children().size());
              for (size_t i = 0; i < tuple.size(); i++) {
                treeFlatten(tuple[i], spec.children(i), leaves);
              }
            },
            [](std::vector<c10::IValue> flats,
               const nlohmann::json& obj) -> c10::IValue {
              TORCH_INTERNAL_ASSERT_DEBUG_ONLY(obj.is_null());
              return c10::ivalue::Tuple::create(std::move(flats));
            },
            [](TreeMapNoReturnFn fn,
               const c10::IValue& tree,
               const TreeSpec& spec) {
              const auto& tuple = tree.toTupleRef().elements();
              TORCH_CHECK_EQ(tuple.size(), spec.children().size());
              for (size_t i = 0; i < tuple.size(); i++) {
                leafApply(fn, tuple[i], spec.children(i));
              }
            }});
    const auto& tupleNodeDef = getNodeDef("builtins.tuple");
    registerNode(
        "collections.namedtuple",
        NodeDef{
            tupleNodeDef.flattenFn,
            [](std::vector<c10::IValue> flats,
               const nlohmann::json& obj) -> c10::IValue {
              TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!obj.is_null());
              return c10::ivalue::Tuple::create(std::move(flats));
            },
            tupleNodeDef.leafApplyFn,
            [](std::string_view context) { return nlohmann::json{context}; }});
    registerNode(
        "builtins.list",
        NodeDef{
            [](const c10::IValue& tree,
               const TreeSpec& spec,
               std::vector<c10::IValue>& leaves) {
              auto list = tree.toList();
              for (size_t i = 0; i < list.size(); i++) {
                treeFlatten(list[i], spec.children(i), leaves);
              }
            },
            [](std::vector<c10::IValue> flats,
               const nlohmann::json& obj) -> c10::IValue {
              TORCH_INTERNAL_ASSERT_DEBUG_ONLY(obj.is_null());
              c10::List<c10::IValue> list(c10::AnyType::get());
              list.reserve(flats.size());
              for (auto& flat : flats) {
                list.push_back(std::move(flat));
              }
              return list;
            },
            [](TreeMapNoReturnFn fn,
               const c10::IValue& tree,
               const TreeSpec& spec) {
              auto list = tree.toList();
              for (size_t i = 0; i < list.size(); i++) {
                leafApply(fn, list[i], spec.children(i));
              }
            }});
    registerNode(
        "torch.fx.immutable_collections.immutable_list",
        getNodeDef("builtins.list"));
    registerNode(
        "builtins.dict",
        NodeDef{
            [](const c10::IValue& tree,
               const TreeSpec& spec,
               std::vector<c10::IValue>& leaves) {
              auto dict = tree.toGenericDict();
              const auto& context = spec.context();
              TORCH_CHECK_EQ(dict.size(), context.size());
              size_t i = 0;
              for (const auto& keyObj : context) {
                auto key = dynamicToIValue(keyObj);
                auto it = dict.find(key);

                if (it != dict.end()) {
                  treeFlatten(it->value(), spec.children(i), leaves);
                } else {
                  // when we have a dict with missing keys, we fill the missing
                  // leaves with c10::IValue()
                  for (size_t j = 0; j < spec.children(i).numLeaves(); ++j) {
                    leaves.emplace_back();
                  }
                }
                i++;
              }
            },
            [](std::vector<c10::IValue> flats,
               const nlohmann::json& obj) -> c10::IValue {
              c10::Dict<c10::IValue, c10::IValue> dict(
                  c10::AnyType::get(), c10::AnyType::get());
              TORCH_CHECK(obj.is_array());
              TORCH_CHECK_EQ(obj.size(), flats.size());
              dict.reserve(flats.size());
              for (size_t i = 0; i < flats.size(); i++) {
                dict.insert(dynamicToIValue(obj[i]), std::move(flats[i]));
              }
              return dict;
            },
            [](TreeMapNoReturnFn fn,
               const c10::IValue& tree,
               const TreeSpec& spec) {
              auto dict = tree.toGenericDict();
              const auto& context = spec.context();

              TORCH_CHECK(
                  dict.size() <= context.size(),
                  "input dict has more keys than treeSepc");

              size_t i = 0;
              for (const auto& keyObj : context) {
                auto key = dynamicToIValue(keyObj);
                auto it = dict.find(key);
                if (it != dict.end()) {
                  leafApply(fn, it->value(), spec.children(i));
                } else {
                  // when we have a dict with missing keys, we run fn
                  // on leaves with value of c10::IValue()
                  for (size_t j = 0; j < spec.children(i).numLeaves(); ++j) {
                    fn(c10::IValue());
                  }
                }
                i++;
              }
            }});
    registerNode(
        "torch.fx.immutable_collections.immutable_dict",
        getNodeDef("builtins.dict"));
  }
  bool hasNodeDef(std::string_view typeName) const {
    return registry_.find(std::string{typeName}) != registry_.end();
  }
  const NodeDef& getNodeDef(std::string_view typeName) const {
    return registry_.at(std::string{typeName});
  }
  void registerNode(std::string_view typeName, NodeDef nodeDef) {
    TORCH_CHECK(!hasNodeDef(typeName));
    registry_.emplace(typeName, std::move(nodeDef));
  }

 private:
  std::unordered_map<std::string, NodeDef> registry_;
};

c10::Synchronized<PytreeNodeRegistry>& getPytreeNodeRegistry() {
  static auto* registry = new c10::Synchronized<PytreeNodeRegistry>();
  return *registry;
}

TreeSpec makeTreeSpec(const nlohmann::json& obj) {
  TORCH_CHECK(obj.is_object());
  TORCH_CHECK(obj.find("type") != obj.end());
  if (obj["type"].is_null()) {
    TORCH_CHECK_EQ(obj["children_spec"].size(), 0);
    TORCH_CHECK(obj["context"].is_null());
    return TreeSpec{};
  }
  const auto& name = obj["type"].get<std::string>();
  NodeDef nodeDefCache;
  getPytreeNodeRegistry().withLock([&](auto& registry) {
    TORCH_CHECK(registry.hasNodeDef(name), "Unknown pytree node type: ", name);
    nodeDefCache = registry.getNodeDef(name);
  });
  auto context = nodeDefCache.contextLoadFn(obj["context"].get<std::string>());
  const auto& childrenSpec = obj["children_spec"];
  TORCH_CHECK(childrenSpec.is_array());
  std::vector<TreeSpec> children;
  for (const auto& child : childrenSpec) {
    children.push_back(makeTreeSpec(child));
  }
  return TreeSpec(name, context, std::move(children), std::move(nodeDefCache));
}

} // namespace

void registerPytreeNode(std::string_view typeName, NodeDef nodeDef) {
  getPytreeNodeRegistry().withLock([&](auto& registry) {
    registry.registerNode(typeName, std::move(nodeDef));
  });
}

TreeSpec treeSpecLoads(std::string_view json) {
  const auto obj = nlohmann::json::parse(json);
  TORCH_CHECK(obj.is_array());
  TORCH_CHECK_EQ(obj.size(), 2);
  TORCH_CHECK_EQ(obj[0].get<int64_t>(), kDefaultTreeSpecSerializationProtocol);
  return makeTreeSpec(obj[1]);
}

c10::IValue treeUnflatten(
    std::vector<c10::IValue> leaves,
    const TreeSpec& spec) {
  RecordFunction recordFunction("nativert::treeUnflatten");

  TORCH_CHECK_EQ(leaves.size(), spec.numLeaves());
  if (spec.isLeaf()) {
    return std::move(leaves[0]);
  }
  auto unflattenFn = spec.nodeDefCache().unflattenFn;
  if (spec.allLeaves()) {
    return unflattenFn(std::move(leaves), spec.context());
  }
  size_t start = 0;
  std::vector<c10::IValue> childrenPytrees;
  for (const auto& child : spec.children()) {
    if (child.isLeaf()) {
      childrenPytrees.push_back(std::move(leaves[start]));
      start++;
      continue;
    }
    size_t numLeaves = child.numLeaves();
    std::vector<c10::IValue> slice(
        std::make_move_iterator(leaves.begin() + start),
        std::make_move_iterator(leaves.begin() + start + numLeaves));
    childrenPytrees.push_back(treeUnflatten(std::move(slice), child));
    start += numLeaves;
  }
  return unflattenFn(std::move(childrenPytrees), spec.context());
}

std::vector<c10::IValue> treeFlatten(
    const c10::IValue& tree,
    const TreeSpec& spec) {
  std::vector<c10::IValue> leaves;
  leaves.reserve(spec.numLeaves());
  treeFlatten(tree, spec, leaves);
  return leaves;
}

std::vector<c10::IValue> treeFlattenFromArgs(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const TreeSpec& spec) {
  RecordFunction recordFunction("nativert::treeFlattenFromArgs");

  TORCH_CHECK(!spec.isLeaf());
  TORCH_CHECK_EQ(spec.children().size(), 2);

  std::vector<c10::IValue> leaves;
  leaves.reserve(spec.numLeaves());
  const auto& specArgs = spec.children(0);
  TORCH_CHECK(!specArgs.isLeaf());
  TORCH_CHECK_EQ(specArgs.children().size(), args.size());
  for (size_t i = 0; i < args.size(); i++) {
    treeFlatten(args[i], specArgs.children(i), leaves);
  }

  const auto& specKwargs = spec.children(1);
  TORCH_CHECK(!specKwargs.isLeaf());
  TORCH_CHECK_EQ(specKwargs.context().size(), kwargs.size());
  for (size_t i = 0; i < specKwargs.context().size(); i++) {
    treeFlatten(
        kwargs.at(specKwargs.context()[i].get<std::string>()),
        specKwargs.children(i),
        leaves);
  }
  return leaves;
}

void leafApplyFromArgs(
    TreeMapNoReturnFn fn,
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const TreeSpec& spec) {
  RecordFunction recordFunction("nativert::leafApplyFromArgs");

  TORCH_CHECK(!spec.isLeaf());
  TORCH_CHECK_EQ(spec.children().size(), 2);

  std::vector<c10::IValue> leaves;
  leaves.reserve(spec.numLeaves());
  const auto& specArgs = spec.children(0);
  TORCH_CHECK(!specArgs.isLeaf());
  TORCH_CHECK_EQ(specArgs.children().size(), args.size());
  for (size_t i = 0; i < args.size(); i++) {
    leafApply(fn, args[i], specArgs.children(i));
  }

  const auto& specKwargs = spec.children(1);
  TORCH_CHECK(!specKwargs.isLeaf());
  TORCH_CHECK_EQ(specKwargs.context().size(), kwargs.size());
  for (size_t i = 0; i < specKwargs.context().size(); i++) {
    leafApply(
        fn,
        kwargs.at(specKwargs.context()[i].get<std::string>()),
        specKwargs.children(i));
  }
}

std::vector<at::Tensor> treeFlattenToTensorList(
    const c10::IValue& tree,
    const TreeSpec& spec) {
  auto flats = treeFlatten(tree, spec);
  std::vector<at::Tensor> tensors;
  tensors.reserve(flats.size());
  for (const auto& flat : flats) {
    tensors.push_back(flat.toTensor());
  }
  return tensors;
}

c10::IValue
treeMap(TreeMapFn f, const c10::IValue& tree, const TreeSpec& spec) {
  const auto flats = treeFlatten(tree, spec);
  std::vector<c10::IValue> mapped;
  mapped.reserve(flats.size());
  for (const auto& flat : flats) {
    mapped.push_back(f(flat));
  }
  return treeUnflatten(std::move(mapped), spec);
}

c10::IValue argsToIValue(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  c10::Dict<c10::IValue, c10::IValue> dict(
      c10::StringType::get(), c10::AnyType::get());
  for (const auto& [key, arg] : kwargs) {
    dict.insert(key, arg);
  }
  return c10::ivalue::Tuple::create({c10::ivalue::Tuple::create(args), dict});
}

std::
    pair<std::vector<c10::IValue>, std::unordered_map<std::string, c10::IValue>>
    treeMapArgs(
        TreeMapFn f,
        const std::vector<c10::IValue>& args,
        const std::unordered_map<std::string, c10::IValue>& kwargs,
        const TreeSpec& spec) {
  const auto val = argsToIValue(args, kwargs);
  const auto mapVal = treeMap(f, val, spec);
  auto mapArgs =
      mapVal.toTupleRef().elements()[0].toTupleRef().elements().vec();
  std::unordered_map<std::string, c10::IValue> mapKwargs;
  for (const auto& entry : mapVal.toTupleRef().elements()[1].toGenericDict()) {
    mapKwargs.emplace(entry.key().toStringRef(), entry.value());
  }
  return {std::move(mapArgs), std::move(mapKwargs)};
}

void leafApply(
    TreeMapNoReturnFn fn,
    const c10::IValue& tree,
    const TreeSpec& spec) {
  if (spec.isLeaf()) {
    fn(tree);
    return;
  }
  auto leafApplyFn = spec.nodeDefCache().leafApplyFn;
  leafApplyFn(fn, tree, spec);
}

nlohmann::json defaultContextLoadFn(std::string_view context) {
  return nlohmann::json::parse(context);
}

c10::TypePtr TreeSpec::toAtenType() const {
  if (isLeaf()) {
    return c10::AnyType::get();
  } else if (uniformName_ == "builtins.tuple") {
    std::vector<c10::TypePtr> childrenType;
    for (const auto& childrenSpec : children_) {
      childrenType.emplace_back(childrenSpec.toAtenType());
    }
    return c10::TupleType::create(std::move(childrenType));
  } else if (
      uniformName_ == "builtins.list" ||
      uniformName_ == "torch.fx.immutable_collections.immutable_list") {
    if (children_.empty()) {
      return c10::ListType::create(c10::AnyType::get());
    } else {
      return c10::ListType::create(children_[0].toAtenType());
    }
  } else if (
      uniformName_ == "builtins.dict" ||
      uniformName_ == "torch.fx.immutable_collections.immutable_dict") {
    if (children_.empty()) {
      return c10::DictType::create(c10::AnyType::get(), c10::AnyType::get());
    } else {
      return c10::DictType::create(
          dynamicToIValue(context_[0]).type(), children_[0].toAtenType());
    }
  } else {
    TORCH_CHECK(false, "Unsupported uniform name: ", uniformName_.value());
  }
}

} // namespace torch::nativert
