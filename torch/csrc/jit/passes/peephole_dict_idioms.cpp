#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/peephole_dict_idioms.h>

namespace torch {
namespace jit {

namespace {

class DictNodeImplBase {
 public:
  virtual ~DictNodeImplBase() = default;

  virtual bool contains(const IValue&) const = 0;
  virtual size_t size() const = 0;
  virtual Value* get(const IValue&) const = 0;

  bool canOptimize() {
    return !has_overlap_ && !has_non_const_key_;
  }

 protected:
  bool has_overlap_ = false;
  bool has_non_const_key_ = false;
};

template <class KeyType>
class DictNodeImpl : public DictNodeImplBase {
 public:
  DictNodeImpl(
      std::function<KeyType(const IValue&)> ivalue_converter,
      Node* dict_creation_node)
      : ivalue_converter_(std::move(ivalue_converter)) {
    for (size_t i = 0; i < dict_creation_node->inputs().size(); i += 2) {
      auto key_opt = toIValue(dict_creation_node->input(i));

      // Key is not constant if we cannot convert to IValue
      if (key_opt == c10::nullopt) {
        has_non_const_key_ = true;
        continue;
      }

      KeyType key = ivalue_converter_(*key_opt);
      if (dict_.find(key) == dict_.end()) {
        dict_.emplace(key, dict_creation_node->input(i + 1));
      } else {
        has_overlap_ = true;
      }
    }
  }

  bool contains(const IValue& ivalue) const override {
    auto key = ivalue_converter_(ivalue);
    return dict_.find(key) != dict_.end();
  }

  size_t size() const override {
    return dict_.size();
  }

  Value* get(const IValue& ivalue) const override {
    auto val = ivalue_converter_(ivalue);
    auto loc = dict_.find(val);
    if (loc != dict_.end()) {
      return loc->second;
    }
    TORCH_CHECK(false, "Cannot get non-existent key");
  }

 private:
  std::unordered_map<KeyType, Value*> dict_;
  std::function<KeyType(const IValue&)> ivalue_converter_;
};

class DictNode {
 public:
  explicit DictNode(Node* dict_creation_node) {
    auto dict_type = dict_creation_node->output()->type();
    auto key_value_types = dict_type->containedTypes();
    TORCH_CHECK(
        key_value_types.size() == 2, "Dict must have 2 contained types");
    const auto& key_type = key_value_types[0];

    switch (key_type->kind()) {
      case TypeKind::IntType: {
        auto ivalue_converter = [](const IValue& ival) { return ival.toInt(); };
        impl_ = std::make_unique<DictNodeImpl<int64_t>>(
            std::move(ivalue_converter), dict_creation_node);
        break;
      }

      case TypeKind::FloatType: {
        auto ivalue_converter = [](const IValue& ival) {
          return ival.toDouble();
        };
        impl_ = std::make_unique<DictNodeImpl<double>>(
            std::move(ivalue_converter), dict_creation_node);
        break;
      }

      case TypeKind::StringType: {
        auto ivalue_converter = [](const IValue& ival) {
          return *ival.toString();
        };
        impl_ = std::make_unique<DictNodeImpl<std::string>>(
            std::move(ivalue_converter), dict_creation_node);
        break;
      }

      default:
        impl_ = nullptr;
    }
  }

  bool canOptimize() const {
    if (impl_) {
      return impl_->canOptimize();
    }
    return false;
  }

  size_t size() const {
    if (impl_) {
      return impl_->size();
    }
    return 0;
  }

  c10::optional<Value*> getOrNullopt(const IValue& key) const {
    if (impl_ && impl_->contains(key)) {
      return impl_->get(key);
    }
    return c10::nullopt;
  }

 private:
  std::unique_ptr<DictNodeImplBase> impl_;
};

bool isDict(Value* v) {
  return v->type()->castRaw<DictType>() != nullptr;
}

class PeepholeOptimizeDictIdiomsImpl {
 public:
  explicit PeepholeOptimizeDictIdiomsImpl(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)), aliasDb_(std::make_unique<AliasDb>(graph_)) {}

  bool run() {
    collectMutatedDicts(graph_->block());
    return runBlock(graph_->block());
  }

 private:
  void checkForMutatedDicts(Value* v) {
    if (isDict(v) && aliasDb_->hasWriters(v)) {
      mutated_dicts_.insert(v);
    }
  }

  void collectMutatedDicts(Block* b) {
    for (Value* v : b->inputs()) {
      checkForMutatedDicts(v);
    }
    for (Node* n : b->nodes()) {
      for (Value* v : n->outputs()) {
        checkForMutatedDicts(v);
      }
      for (Block* block : n->blocks()) {
        collectMutatedDicts(block);
      }
    }
  }

  const DictNode& getDictNode(Node* creation_node) {
    auto cached = dict_cache_.find(creation_node);
    if (cached == dict_cache_.end()) {
      cached =
          dict_cache_.emplace(creation_node, DictNode(creation_node)).first;
    }

    return cached->second;
  }

  c10::optional<Value*> getValueFromDict(Node* dict_creation_node, Value* key) {
    const DictNode& dict_node = getDictNode(dict_creation_node);
    auto key_opt = toIValue(key);
    // Key is not constant if we cannot convert to IValue
    if (key_opt == c10::nullopt) {
      return c10::nullopt;
    }
    IValue key_ival = *key_opt;
    if (dict_node.canOptimize()) {
      return dict_node.getOrNullopt(key_ival);
    }
    return c10::nullopt;
  }

  c10::optional<int64_t> computeLen(Node* dict_creation_node) {
    const DictNode& dict_node = getDictNode(dict_creation_node);
    if (dict_node.canOptimize()) {
      return static_cast<int64_t>(dict_node.size());
    }
    return c10::nullopt;
  }

  bool optimizeLen(Node* len_node, Node* creation_node) {
    if (creation_node->kind() == prim::DictConstruct) {
      auto len = computeLen(creation_node);
      if (len != c10::nullopt) {
        WithInsertPoint guard(len_node);
        len_node->output()->replaceAllUsesWith(graph_->insertConstant(len));
        return true;
      }
    }
    return false;
  }

  bool optimizeGetItem(Node* getitem_node, Node* creation_node) {
    if (creation_node->kind() == prim::DictConstruct) {
      auto key = getitem_node->input(1);
      auto value = getValueFromDict(creation_node, key);
      if (value != c10::nullopt) {
        getitem_node->output()->replaceAllUsesWith(*value);
        return true;
      }
    }
    return false;
  }

  bool runBlock(Block* block) {
    bool changed = false;
    for (Node* node : block->nodes()) {
      for (Block* b : node->blocks()) {
        changed |= runBlock(b);
      }

      // only optimizing dict ops
      if (node->inputs().empty() || !isDict(node->input(0))) {
        continue;
      }

      auto first_input = node->input(0);

      // only optimizing ops with unmutated inputs
      if (mutated_dicts_.count(first_input)) {
        continue;
      }

      if (node->kind() == aten::len) {
        changed |= optimizeLen(node, first_input->node());
      } else if (node->kind() == aten::__getitem__) {
        changed |= optimizeGetItem(node, first_input->node());
      }
    }
    return changed;
  }

  std::shared_ptr<Graph> graph_;
  std::unordered_set<Value*> mutated_dicts_;
  std::unique_ptr<AliasDb> aliasDb_;
  std::unordered_map<Node*, DictNode> dict_cache_;
};

} // namespace

bool PeepholeOptimizeDictIdioms(const std::shared_ptr<Graph>& graph) {
  PeepholeOptimizeDictIdiomsImpl opt(graph);
  return opt.run();
}

} // namespace jit
} // namespace torch
