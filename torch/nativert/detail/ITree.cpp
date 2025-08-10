#include <ATen/record_function.h>
#include <torch/nativert/detail/ITree.h>

#include <iterator>
#include <string_view>

#include <ATen/core/ivalue.h>
#include <c10/util/Synchronized.h>
#include <nlohmann/json.hpp>

namespace torch::nativert::detail {

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

void itreeFlatten(
    const c10::IValue& nested,
    const ITreeSpec& spec,
    std::vector<c10::IValue>& ivalues) {
  if (spec.isIValue()) {
    ivalues.push_back(nested);
    return;
  }
  auto flattenFn = spec.nodeDefCache().flattenFn;
  flattenFn(nested, spec, ivalues);
}

class PytreeNodeRegistry {
 public:
  PytreeNodeRegistry() {
    // Add some law of physics here.
    registerNode(
        "builtins.tuple",
        NodeDef{
            [](const c10::IValue& nested,
               const ITreeSpec& spec,
               std::vector<c10::IValue>& ivalues) {
              const auto& tuple = nested.toTupleRef().elements();
              TORCH_CHECK(tuple.size() == spec.children().size());
              for (size_t i = 0; i < tuple.size(); i++) {
                itreeFlatten(tuple[i], spec.children(i), ivalues);
              }
            },
            [](std::vector<c10::IValue> flats,
               const nlohmann::json& obj) -> c10::IValue {
              TORCH_INTERNAL_ASSERT_DEBUG_ONLY(obj.is_null());
              return c10::ivalue::Tuple::create(std::move(flats));
            },
            [](ITreeMapNoReturnFn fn,
               const c10::IValue& nested,
               const ITreeSpec& spec) {
              const auto& tuple = nested.toTupleRef().elements();
              TORCH_CHECK(tuple.size() == spec.children().size());
              for (size_t i = 0; i < tuple.size(); i++) {
                ivalueApply(fn, tuple[i], spec.children(i));
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
            tupleNodeDef.ivalueApplyFn,
            [](std::string_view context) { return nlohmann::json{context}; }});
    registerNode(
        "builtins.list",
        NodeDef{
            [](const c10::IValue& nested,
               const ITreeSpec& spec,
               std::vector<c10::IValue>& ivalues) {
              auto list = nested.toListRef();
              for (size_t i = 0; i < list.size(); i++) {
                itreeFlatten(list[i], spec.children(i), ivalues);
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
            [](ITreeMapNoReturnFn fn,
               const c10::IValue& nested,
               const ITreeSpec& spec) {
              auto list = nested.toListRef();
              for (size_t i = 0; i < list.size(); i++) {
                ivalueApply(fn, list[i], spec.children(i));
              }
            }});
    registerNode(
        "torch.fx.immutable_collections.immutable_list",
        getNodeDef("builtins.list"));
    registerNode(
        "builtins.dict",
        NodeDef{
            [](const c10::IValue& nested,
               const ITreeSpec& spec,
               std::vector<c10::IValue>& ivalues) {
              auto dict = nested.toGenericDict();
              const auto& contextKeys = spec.contextKeys();
              // allow the dict size less than the spec, missing key will be
              // filled with empty tensor
              TORCH_CHECK(dict.size() <= contextKeys.size());
              size_t i = 0;
              for (const auto& key : contextKeys) {
                auto it = dict.find(key);

                if (it != dict.end()) {
                  itreeFlatten(it->value(), spec.children(i), ivalues);
                } else {
                  // when we have a dict with missing keys, we fill the missing
                  // ivalues with an empty tensor which is required for
                  // validation
                  for (size_t j = 0; j < spec.children(i).numIValues(); ++j) {
                    at::Tensor empty_tensor;
                    ivalues.emplace_back(std::move(empty_tensor));
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
              TORCH_CHECK(obj.size() == flats.size());
              dict.reserve(flats.size());
              for (size_t i = 0; i < flats.size(); i++) {
                dict.insert(dynamicToIValue(obj[i]), std::move(flats[i]));
              }
              return dict;
            },
            [](ITreeMapNoReturnFn fn,
               const c10::IValue& nested,
               const ITreeSpec& spec) {
              auto dict = nested.toGenericDict();
              const auto& contextKeys = spec.contextKeys();

              size_t i = 0;
              for (const auto& key : contextKeys) {
                if (spec.children(i).isUsed()) {
                  auto it = dict.find(key);
                  if (it != dict.end()) {
                    ivalueApply(fn, it->value(), spec.children(i));
                  } else {
                    TORCH_CHECK(false, "input arg is missing key ", key);
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
    registry_.emplace(typeName, nodeDef);
  }

 private:
  std::unordered_map<std::string, NodeDef> registry_;
};

c10::Synchronized<PytreeNodeRegistry>& getPytreeNodeRegistry() {
  static auto* registry = new c10::Synchronized<PytreeNodeRegistry>();
  return *registry;
}

ITreeSpec makeITreeSpec(
    const nlohmann::json& obj,
    const std::vector<const Value*>& values,
    int start) {
  TORCH_CHECK(obj.is_object());
  TORCH_CHECK(obj.find("type") != obj.end());
  if (obj["type"].is_null()) {
    TORCH_CHECK(obj["children_spec"].empty());
    TORCH_CHECK(obj["context"].is_null());

    const Value* value = values[start];
    if (value) {
      bool isUsed = !value->users().empty();
      return ITreeSpec(value, isUsed);
    } else {
      return ITreeSpec(value, false);
    }
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
  std::vector<ITreeSpec> children;
  int offset = 0;
  for (const auto& child : childrenSpec) {
    children.push_back(makeITreeSpec(child, values, start + offset));
    // NOLINTNEXTLINE(*-narrowing-conversions)
    offset += children.back().numIValues();
  }

  return ITreeSpec(name, context, std::move(children), nodeDefCache);
}

} // namespace

void registerPytreeNode(std::string_view typeName, NodeDef nodeDef) {
  getPytreeNodeRegistry().withLock([&](auto& registry) {
    registry.registerNode(typeName, std::move(nodeDef));
  });
}

ITreeSpec itreeSpecLoads(
    std::string_view json,
    const std::vector<const Value*>& values) {
  const auto obj = nlohmann::json::parse(json);
  TORCH_CHECK(obj.is_array());
  TORCH_CHECK(obj.size() == 2);
  TORCH_CHECK(obj[0].get<int64_t>() == kDefaultTreeSpecSerializationProtocol);
  auto result = makeITreeSpec(obj[1], values, 0);

  TORCH_CHECK(result.numIValues() == values.size());
  return result;
}

c10::IValue itreeUnflatten(
    std::vector<c10::IValue> ivalues,
    const ITreeSpec& spec) {
  RECORD_USER_SCOPE("nativert::itreeUnflatten");
  TORCH_CHECK(ivalues.size() == spec.numIValues());
  if (spec.isIValue()) {
    return std::move(ivalues[0]);
  }
  auto unflattenFn = spec.nodeDefCache().unflattenFn;
  if (spec.allIValues()) {
    return unflattenFn(std::move(ivalues), spec.context());
  }
  size_t start = 0;
  std::vector<c10::IValue> childrenPytrees;
  for (const auto& child : spec.children()) {
    if (child.isIValue()) {
      childrenPytrees.push_back(std::move(ivalues[start]));
      start++;
      continue;
    }
    size_t numIValues = child.numIValues();
    std::vector<c10::IValue> slice(
        // NOLINTNEXTLINE(*-narrowing-conversions)
        std::make_move_iterator(ivalues.begin() + start),
        // NOLINTNEXTLINE(*-narrowing-conversions)
        std::make_move_iterator(ivalues.begin() + start + numIValues));
    childrenPytrees.push_back(itreeUnflatten(std::move(slice), child));
    start += numIValues;
  }
  return unflattenFn(std::move(childrenPytrees), spec.context());
}

std::vector<c10::IValue> itreeFlatten(
    const c10::IValue& nested,
    const ITreeSpec& spec) {
  std::vector<c10::IValue> ivalues;
  ivalues.reserve(spec.numIValues());
  itreeFlatten(nested, spec, ivalues);
  return ivalues;
}

std::vector<c10::IValue> itreeFlattenFromArgs(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const ITreeSpec& spec) {
  RECORD_USER_SCOPE("nativert::itreeFlattenFromArgs");
  TORCH_CHECK(!spec.isIValue());
  TORCH_CHECK(spec.children().size() == 2);

  std::vector<c10::IValue> ivalues;
  ivalues.reserve(spec.numIValues());
  const auto& specArgs = spec.children(0);
  TORCH_CHECK(!specArgs.isIValue());
  TORCH_CHECK(specArgs.children().size() == args.size());
  for (size_t i = 0; i < args.size(); i++) {
    itreeFlatten(args[i], specArgs.children(i), ivalues);
  }

  const auto& specKwargs = spec.children(1);
  TORCH_CHECK(!specKwargs.isIValue());
  TORCH_CHECK(specKwargs.context().size() == kwargs.size());
  for (size_t i = 0; i < specKwargs.context().size(); i++) {
    itreeFlatten(
        kwargs.at(specKwargs.context()[i].get_ref<const std::string&>()),
        specKwargs.children(i),
        ivalues);
  }
  return ivalues;
}

void ivalueApplyFromArgs(
    ITreeMapNoReturnFn fn,
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const ITreeSpec& spec) {
  RECORD_USER_SCOPE("nativert::ivalueApplyFromArgs");
  TORCH_CHECK(!spec.isIValue());
  TORCH_CHECK(spec.children().size() == 2);

  const auto& specArgs = spec.children(0);
  TORCH_CHECK(!specArgs.isIValue());
  TORCH_CHECK(specArgs.children().size() == args.size());
  for (size_t i = 0; i < args.size(); i++) {
    ivalueApply(fn, args[i], specArgs.children(i));
  }

  const auto& specKwargs = spec.children(1);
  TORCH_CHECK(!specKwargs.isIValue());

  const auto& ctx = specKwargs.context();
  TORCH_CHECK(ctx.size() == kwargs.size());

  for (size_t i = 0; i < ctx.size(); i++) {
    ivalueApply(
        fn,
        kwargs.at(ctx[i].get_ref<const std::string&>()),
        specKwargs.children(i));
  }
}

std::vector<at::Tensor> itreeFlattenToTensorList(
    const c10::IValue& nested,
    const ITreeSpec& spec) {
  auto flats = itreeFlatten(nested, spec);
  std::vector<at::Tensor> tensors;
  tensors.reserve(flats.size());
  for (const auto& flat : flats) {
    tensors.push_back(flat.toTensor());
  }
  return tensors;
}

c10::IValue itreeMap(
    ITreeMapFn f,
    const c10::IValue& nested,
    const ITreeSpec& spec) {
  const auto flats = itreeFlatten(nested, spec);
  std::vector<c10::IValue> mapped;
  mapped.reserve(flats.size());
  for (const auto& flat : flats) {
    mapped.push_back(f(flat));
  }
  return itreeUnflatten(std::move(mapped), spec);
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
    itreeMapArgs(
        ITreeMapFn f,
        const std::vector<c10::IValue>& args,
        const std::unordered_map<std::string, c10::IValue>& kwargs,
        const ITreeSpec& spec) {
  const auto val = argsToIValue(args, kwargs);
  const auto mapVal = itreeMap(f, val, spec);
  auto mapArgs =
      mapVal.toTupleRef().elements()[0].toTupleRef().elements().vec();
  std::unordered_map<std::string, c10::IValue> mapKwargs;
  for (const auto& entry : mapVal.toTupleRef().elements()[1].toGenericDict()) {
    mapKwargs.emplace(entry.key().toStringRef(), entry.value());
  }
  return {std::move(mapArgs), std::move(mapKwargs)};
}

void ivalueApply(
    ITreeMapNoReturnFn fn,
    const c10::IValue& nested,
    const ITreeSpec& spec) {
  if (spec.isIValue()) {
    if (spec.isUsed()) {
      fn(nested, spec.value());
    }
    return;
  }
  auto ivalueApplyFn = spec.nodeDefCache().ivalueApplyFn;
  ivalueApplyFn(fn, nested, spec);
}

nlohmann::json defaultContextLoadFn(std::string_view context) {
  return nlohmann::json::parse(context);
}

ITreeSpec::ITreeSpec(
    std::string_view uniformName,
    nlohmann::json context,
    std::vector<ITreeSpec> children,
    NodeDef nodeDefCache)
    : uniformName_(uniformName),
      context_(std::move(context)),
      children_(std::move(children)),
      nodeDefCache_(nodeDefCache),
      numIValues_(0),
      value_(nullptr),
      isUsed_(false) {
  for (auto& child : children_) {
    numIValues_ += child.numIValues();
    allIValues_ &= child.isIValue();
    isUsed_ |= child.isUsed();
  }

  if (uniformName_ == "builtins.dict" ||
      uniformName_ == "torch.fx.immutable_collections.immutable_dict") {
    for (const auto& keyObj : context_) {
      contextKeys_.push_back(dynamicToIValue(keyObj));
    }
  }
}

c10::TypePtr ITreeSpec::toAtenType() const {
  if (isIValue()) {
    return c10::AnyType::get();
  } else if (uniformName_ == "builtins.tuple") {
    std::vector<c10::TypePtr> childrenType;
    childrenType.reserve(children_.size());
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
    TORCH_CHECK(false, "Unsupported uniform name: ", uniformName());
  }
}

} // namespace torch::nativert::detail
