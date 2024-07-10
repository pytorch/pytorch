#include <torch/csrc/jit/frontend/sugared_value.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/frontend/tree_views.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>

namespace torch::jit {

struct NoneValue : SugaredValue {
  NoneValue() = default;
  std::string kind() const override {
    return "None";
  }
};

std::shared_ptr<SugaredValue> PrintValue::call(
    const SourceRange& loc,
    GraphFunction& m,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    size_t n_binders) {
  auto& g = *m.graph();
  if (!kwargs.empty())
    throw ErrorReport(loc) << "print doesn't accept any keyword arguments";

  std::vector<Value*> lowered_inputs = toValues(*m.graph(), args);
  g.insertNode(g.create(prim::Print, lowered_inputs, 0)->setSourceRange(loc));
  return std::make_shared<NoneValue>();
}

static const std::unordered_map<std::string, at::ScalarType>&
builtin_cast_method_to_scalar_type() {
  static std::unordered_map<std::string, at::ScalarType> mapping = {
      {"byte", at::kByte},
      {"char", at::kChar},
      {"double", at::kDouble},
      {"float", at::kFloat},
      {"cfloat", at::kComplexFloat},
      {"cdouble", at::kComplexDouble},
      {"int", at::kInt},
      {"long", at::kLong},
      {"short", at::kShort},
      {"half", at::kHalf}};
  return mapping;
}

std::shared_ptr<SugaredValue> BuiltinFunction::call(
    const SourceRange& loc,
    GraphFunction& m,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    size_t n_binders) {
  return std::make_shared<SimpleValue>(
      emitBuiltinCall(loc, *m.graph(), symbol, args, kwargs, self));
}

// older versions of gcc/clang have a bug where enums can't be used as keys
// in a map by default
// https://stackoverflow.com/questions/18837857/cant-use-enum-class-as-unordered-map-key
struct EnumClassHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

bool SimpleValue::hasAttr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) {
  if (auto class_type = value_->type()->cast<ClassType>()) {
    return class_type->hasMethod(field) || class_type->hasAttribute(field) ||
        class_type->hasConstant(field);
  } else if (auto tuple_type = value_->type()->cast<TupleType>()) {
    if (tuple_type->schema()) {
      for (const auto& arg : tuple_type->schema()->arguments()) {
        if (arg.name() == field) {
          return true;
        }
      }
      return false;
    } else {
      throw ErrorReport(loc) << "hasattr's first argument must be a object "
                             << "or NamedTuple, but got a normal Tuple "
                             << value_->type()->repr_str() << " instead";
    }
  }
  throw ErrorReport(loc) << "hasattr's first argument must be an object or "
                         << "NamedTuple, got " << value_->type()->repr_str()
                         << " instead";
}

// support syntax sugar for x.foo(y, z) by allowing x.foo to return a
// callable value that will resolve to foo(x, y, z) when called.
std::shared_ptr<SugaredValue> SimpleValue::attr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) {
  // Allow method-style casts on Tensor types. e.g. x.int()
  if (value_->type()->isSubtypeOf(*TensorType::get())) {
    if (builtin_cast_method_to_scalar_type().count(field)) {
      return std::make_shared<TensorCastValue>(
          builtin_cast_method_to_scalar_type().at(field),
          NamedValue(loc, "self", value_));
    }
  }
  // accessing properties of Tensor and Device that are implemented as
  // prim:: or aten:: operators
  using PropertiesLookup = std::unordered_map<
      TypeKind,
      std::unordered_map<std::string, std::string>,
      EnumClassHash>;
  static const PropertiesLookup builtin_properties = {
      {TypeKind::OptionalType,
       {
           {"unchecked_unwrap_optional", "prim"},
       }},
      {TypeKind::TensorType,
       {
           {"dtype", "prim"},
           {"device", "prim"},
           {"grad", "prim"},
           {"data", "prim"},
           {"shape", "prim"},
           {"is_cuda", "prim"},
           {"is_cpu", "prim"},
           {"is_xla", "prim"},
           {"is_xpu", "prim"},
           {"is_sparse", "prim"},
           {"is_sparse_csr", "prim"},
           {"is_mkldnn", "prim"},
           {"is_mps", "prim"},
           {"is_mtia", "prim"},
           {"is_quantized", "prim"},
           {"is_vulkan", "prim"},
           {"is_ipu", "prim"},
           {"is_meta", "prim"},
           {"is_leaf", "aten"},
           {"is_nested", "prim"},
           {"requires_grad", "prim"},
           {"layout", "prim"},
           {"T", "prim"},
           {"H", "prim"},
           {"mT", "aten"},
           {"mH", "aten"},
           {"is_maia", "prim"},
           {"itemsize", "prim"},
           {"nbytes", "prim"},
           {"ndim", "prim"},
           {"name", "prim"},
           {"real", "aten"},
           {"imag", "aten"},
           {"retains_grad", "aten"},
       }},
      {TypeKind::DeviceObjType, {{"type", "prim"}, {"index", "prim"}}}};
  auto kind = value_->type()->kind();
  auto types_for_builtin = builtin_properties.find(kind);
  if (types_for_builtin != builtin_properties.end()) {
    auto builtin_entry = types_for_builtin->second.find(field);
    if (builtin_entry != types_for_builtin->second.end()) {
      // A builtin was found, add it to the graph
      auto the_namespace = builtin_entry->second;
      auto r = m.graph()->insert(
          Symbol::fromQualString(the_namespace + "::" + field), {value_});
      return std::make_shared<SimpleValue>(r);
    }
  }

  // accessing fields of named tuples
  if (auto tuple_type = value_->type()->cast<TupleType>()) {
    if (tuple_type->schema()) {
      auto attrs = tuple_type->schema()->arguments();
      for (const auto i : c10::irange(attrs.size())) {
        if (attrs[i].name() == field) {
          auto idx = m.graph()->insertConstant(IValue(static_cast<int64_t>(i)));
          auto out_type = tuple_type->elements().at(i);
          auto r = m.graph()
                       ->insertNode(
                           m.graph()->createTupleIndex(value_, idx, out_type))
                       ->output();
          return std::make_shared<SimpleValue>(r);
        }
      }
    }
  } else if (auto awaitType = value_->type()->cast<AwaitType>()) {
    auto elType = awaitType->getElementType();
    auto& g = *m.graph();
    auto v = g.insert(prim::awaitable_wait, {value_}, {}, loc);
    auto sv = std::make_shared<SimpleValue>(v);
    return sv->attr(loc, m, field);
  } else if (auto classType = value_->type()->cast<ClassType>()) {
    // This is a class, emit the proper attribute lookup
    if (classType->findMethod(field)) {
      return std::make_shared<MethodValue>(getValue(), field);
    }
    if (classType->hasAttribute(field)) {
      auto& g = *m.graph();
      auto n = g.insertNode(g.createGetAttr(value_, field));
      return std::make_shared<SimpleValue>(n->output());
    }
    // Check and see if it's a getter attribute.
    auto prop = classType->getProperty(field);
    if (prop) {
      return MethodValue(value_, prop->getter->name())
          .call(loc, m, {}, {}, /*n_binders=*/1);
    }
  } else if (auto iface = value_->type()->cast<InterfaceType>()) {
    // accessing methods of interfaces
    if (iface->getMethod(field)) {
      return std::make_shared<MethodValue>(getValue(), field);
    }
  } else if (auto enum_type = value_->type()->cast<EnumType>()) {
    // Handle access to Enum's `name` and `value` attribute.
    auto& g = *m.graph();

    if (field == "name") {
      auto n = g.insertNode(g.createEnumName(value_));
      return std::make_shared<SimpleValue>(n->output());
    }

    if (field == "value") {
      auto n = g.insertNode(g.createEnumValue(value_));
      return std::make_shared<SimpleValue>(n->output());
    }
  }

  // none of the more-specific cases worked, so see if this is a builtin method
  // If field is a type, then call the aten::to op
  if (field == "type") {
    if (auto builtin = BuiltinFunction::tryCreate(
            Symbol::aten("to"), NamedValue(loc, "self", value_))) {
      return builtin;
    }
  }

  if (auto builtin = BuiltinFunction::tryCreate(
          Symbol::aten(field), NamedValue(loc, "self", value_))) {
    return builtin;
  }

  // Handle calling tolist() on a Tensor.
  if (value_->type()->isSubtypeOf(*TensorType::get()) && field == "tolist") {
    return SpecialFormValue::create(prim::tolist);
  }

  // Handle calling __getitem__() directly on a Tensor, it needs special
  // handling because desired method name (`__getitem__`) doesn't match `aten`
  // operator name of `aten::index`.
  if (value_->type()->isSubtypeOf(*TensorType::get()) &&
      field == "__getitem__") {
    return SpecialFormValue::create(aten::index);
  }

  if (auto generator_type = value_->type()->cast<GeneratorType>()) {
    // Handle access to Generator's `manual_seed`, `initial_seed` and `seed`
    // attributes.
    if (field == "manual_seed" || field == "initial_seed" || field == "seed") {
      if (auto builtin = BuiltinFunction::tryCreate(
              Symbol::aten(field), NamedValue(loc, "self", value_))) {
        return builtin;
      }
    }
  }

  ErrorReport report(loc);
  report << "'" << value_->type()->repr_str()
         << "' object has no attribute or method '" << field << "'.";
  if (auto classType = value_->type()->cast<ClassType>()) {
    if (classType->isUnresolvedClassAttribute(field)) {
      report
          << " '" << field
          << "' is defined as a class attribute which currently is not"
             " supported. Consider converting this to an instance attribute.";
    } else {
      report << " Did you forget to initialize an attribute in __init__()?";
    }
  }
  throw report;
}

std::vector<std::shared_ptr<SugaredValue>> SimpleValue::asTuple(
    const SourceRange& loc,
    GraphFunction& m,
    const std::optional<size_t>& size_hint) {
  static const auto make_simple_value =
      [](Value* v) -> std::shared_ptr<SugaredValue> {
    return std::make_shared<SimpleValue>(v);
  };
  if (value_->type()->kind() == TypeKind::TupleType) {
    auto outputs = createTupleUnpack(value_);
    return fmap(outputs, make_simple_value);
  } else if (value_->type()->kind() == TypeKind::ListType) {
    if (!size_hint) {
      throw ErrorReport(loc)
          << "cannot statically infer the expected size of a "
          << "list in this context";
    }
    auto graph = value_->owningGraph();
    Node* unpack =
        graph->insertNode(graph->createListUnpack(value_, *size_hint));
    return fmap(unpack->outputs(), make_simple_value);
  } else if (value_->type()->kind() == TypeKind::AnyTupleType) {
    throw ErrorReport(loc)
        << "Provided tuple is not fully defined/refined including its element types, please provide a value of type like Tuple[int, int]";
  }
  throw ErrorReport(loc) << value_->type()->repr_str()
                         << " cannot be used as a tuple";
}

static bool isRecursive(const TypePtr& classType, const TypePtr& attrType) {
  if (attrType->isSubtypeOf(*classType)) {
    return true;
  }

  // Recursively check contained types. We need to do this because a user may do
  // A -> B -> A.
  for (const auto& type : attrType->containedTypes()) {
    if (isRecursive(classType, type)) {
      return true;
    }
  }
  return false;
}

void SimpleValue::setAttr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field,
    Value* newValue) {
  const auto classType = value_->type()->cast<ClassType>();
  if (!classType) {
    throw ErrorReport(loc) << "Tried to set an attribute: " << field
                           << " on a non-class: " << value_->type()->repr_str();
  }
  auto expectedType = classType->findAttribute(field);
  if (!expectedType) {
    // If we are still compiling the __init__ method for this class, then
    // setting an unknown attribute adds it to the class's definition.

    // We are initializing if:
    const auto isInitializing =
        // 1. The method we're currently inserting into is an init method
        // TODO this can be a qualified name check
        m.name() == "__init__" &&
        // 2. The `self` arg matches this value's type (i.e. we are in the init
        // method for this class, not some other class)
        !m.graph()->inputs().empty() &&
        m.graph()->inputs().at(0)->type() == classType;

    if (isInitializing) {
      if (isRecursive(classType, newValue->type())) {
        throw ErrorReport(loc)
            << "Assignment to attribute '" << field
            << "' cannot be of a type that contains class "
            << "'" << classType->repr_str() << "'.\n"
            << "Classes that recursively contain instances of themselves"
            << " are not yet supported";
      }

      classType->addAttribute(field, newValue->type());
      expectedType = newValue->type();

      const auto insertPoint = m.graph()->insertPoint();
      const auto topLevelBlock = m.graph()->block();
      if (insertPoint->owningBlock() != topLevelBlock) {
        throw ErrorReport(loc)
            << "First assignment cannot be in a control-flow block. "
            << "Initialize the field at the top level first";
      }
    } else {
      // Check and see if it's a setter attribute.
      auto prop = classType->getProperty(field);
      if (prop && prop->setter) {
        MethodValue(value_, prop->setter->name())
            .call(loc, m, {newValue}, {}, /*n_binders=*/1);
        return;
      }

      if (prop && !prop->setter) {
        throw ErrorReport(loc) << "Tried to set read-only attribute: " << field;
      }

      throw ErrorReport(loc)
          << "Tried to set nonexistent attribute: " << field
          << ". Did you forget to initialize it in __init__()?";
    }
  }

  AT_ASSERT(expectedType);

  // Check type correctness
  const auto newType = newValue->type();
  if (!newType->isSubtypeOf(*expectedType)) {
    throw ErrorReport(loc) << "Wrong type for attribute assignment. Expected "
                           << expectedType->repr_str() << " but got "
                           << newType->repr_str();
  }

  auto& g = *m.graph();
  g.insertNode(g.createSetAttr(value_, field, newValue));
}

std::shared_ptr<SugaredValue> SimpleValue::call(
    const SourceRange& loc,
    GraphFunction& m,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    size_t n_binders) {
  // allow our 'fake' closures to be called, used for fork serialization
  // at the moment, but can be expanded later
  Node* self = getValue()->node();
  if (self->kind() == prim::TupleConstruct && self->inputs().size() == 2 &&
      self->inputs().at(0)->node()->kind() == prim::Closure) {
    std::shared_ptr<Graph> graph =
        self->inputs().at(0)->node()->g(attr::Subgraph);
    Value* context = self->inputs().at(1);
    AT_ASSERT(context->node()->kind() == prim::TupleConstruct);

    // fork nodes are emitted in their own block but we do not simplify
    // tuple construction across blocks. To ensure we clean up the tuple
    // construct create another copy of the tuple construct in the fork block
    Value* close_context =
        m.graph()
            ->insertNode(m.graph()->createTuple(context->node()->inputs()))
            ->output();
    // TODO this needs to go in `m`s compilation unit
    auto cu = std::make_shared<CompilationUnit>();
    auto fn = cu->create_function(QualifiedName("anon"), graph);
    auto ret = StrongFunctionPtr(std::move(cu), fn);

    std::vector<NamedValue> ctx_inputs = {close_context};
    ctx_inputs.insert(ctx_inputs.end(), args.begin(), args.end());
    return FunctionValue(ret).call(loc, m, ctx_inputs, kwargs, n_binders);
  }

  if (auto class_type = getValue()->type()->cast<ClassType>()) {
    return attr(loc, m, "__call__")->call(loc, m, args, kwargs, n_binders);
  }

  return SugaredValue::call(loc, m, args, kwargs, n_binders);
}

Value* SimpleValue::len(const SourceRange& loc, GraphFunction& m) {
  // List, Tuple, Tensor, fill in missing information desugaring
  Value* val = getValue();
  TypePtr val_type = val->type();
  Graph& g = *m.graph();
  if (val_type->cast<ListType>() || val_type->cast<StringType>() ||
      val_type->isSubtypeOf(*TensorType::get())) {
    return g.insert(aten::len, {val}, {}, loc);
  } else {
    throw ErrorReport(loc) << "'" << val_type->repr_str() << "'"
                           << " object is not iterable";
  }
}

SugaredValuePtr SimpleValue::getitem(
    const SourceRange& loc,
    GraphFunction& m,
    Value* idx,
    TypePtr type_hint) {
  Value* val = getValue();
  TypePtr val_type = val->type();
  Graph& g = *m.graph();

  // if it's a List/String/Dict, emit a regular __getitem__ op
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (val_type->cast<ListType>() || val_type->cast<StringType>()) {
    return std::make_shared<SimpleValue>(
        g.insert(aten::__getitem__, {val, idx}, {}, loc));
  } else if (auto dict_type = val_type->cast<DictType>()) {
    return std::make_shared<SimpleValue>(
        g.insert(aten::__getitem__, {val, idx}, {}, loc));
  } else if (val_type->isSubtypeOf(*TensorType::get())) {
    return std::make_shared<SimpleValue>(
        g.insert(aten::select, {val, 0, idx}, {}, loc));
  } else if (auto class_type = val_type->cast<ClassType>()) {
    // Check if this is an indexing operation enabled by a type hint.
    // The ModuleDict has already been checked during IR generation to make
    // sure its contents implement the module interface referred to by
    // type_hint.
    if (class_type->is_module() && type_hint) {
      auto res = g.insert(prim::ModuleContainerIndex, {val, idx}, {}, loc);
      res->setType(type_hint);
      return std::make_shared<SimpleValue>(res);
    }

    // Defer to the __getitem__ attr on the class.
    return attr(loc, m, "__getitem__")->call(loc, m, {idx}, {}, 1);
  } else {
    throw ErrorReport(loc) << "'" << val_type->repr_str() << "'"
                           << " object is not subscriptable";
  }
}

SugaredValuePtr SimpleValue::iter(const SourceRange& loc, GraphFunction& m) {
  auto value = getValue();
  auto type = value->type();
  // built-in iterable types
  if (type->cast<ListType>() || type->cast<StringType>() ||
      type->cast<TensorType>()) {
    return std::make_shared<SimpleValue>(value);
  }
  // dicts iterate over keys
  if (type->cast<DictType>()) {
    return std::make_shared<SimpleValue>(
        m.graph()->insert(aten::keys, {value}, {}, loc));
  }
  if (auto tup = type->cast<TupleType>()) {
    auto tup_values = createTupleUnpack(value);
    std::vector<SugaredValuePtr> tup_sugared;
    for (Value* v : tup_values) {
      tup_sugared.push_back(std::make_shared<SimpleValue>(v));
    }
    return std::make_shared<SugaredTupleValue>(tup_sugared);
  } else {
    throw ErrorReport(loc) << "'" << type->repr_str() << "'"
                           << " object is not iterable";
  }
}

RangeValue::RangeValue(
    const SourceRange& loc,
    GraphFunction& m,
    std::vector<Value*> inputs,
    std::optional<int64_t> static_len) {
  for (const auto i : c10::irange(inputs.size())) {
    auto typ = inputs[i]->type();
    if (!typ->cast<IntType>()) {
      throw ErrorReport(loc)
          << "all inputs of range must be ints, found " << typ->repr_str()
          << " in argument " << std::to_string(i);
    }
  }

  Graph& g = *m.graph();
  if (inputs.empty()) {
    throw ErrorReport(loc) << "range expected at least 1 arguments, got 0";
  } else if (inputs.size() == 1) {
    end_ = inputs[0];
    start_ = g.insertConstant(0, loc);
    step_ = g.insertConstant(1, loc);
    // range() call only contains end, easier to calculate len() and getitem()
    has_only_end_ = true;
  } else if (inputs.size() <= 3) {
    start_ = inputs[0];
    end_ = inputs[1];
    if (inputs.size() == 3) {
      step_ = inputs[2];
    } else {
      step_ = g.insertConstant(1, loc);
    }
    has_only_end_ = false;
  } else {
    throw ErrorReport(loc) << "range expected at most 3 arguments, got "
                           << inputs.size();
  }

  static_len_ = static_len;
}

SugaredValuePtr RangeValue::iter(const SourceRange& loc, GraphFunction& m) {
  return shared_from_this();
};

Value* RangeValue::len(const SourceRange& loc, GraphFunction& m) {
  if (static_len_) {
    return insertConstant(*m.graph(), *static_len_, loc);
  }
  if (has_only_end_) {
    return end_;
  } else {
    Graph& g = *m.graph();
    return g.insert(aten::__range_length, {start_, end_, step_}, {}, loc);
  }
}

SugaredValuePtr RangeValue::getitem(
    const SourceRange& loc,
    GraphFunction& m,
    Value* idx,
    TypePtr type_hint) {
  if (has_only_end_) {
    return std::make_shared<SimpleValue>(idx);
  } else {
    auto& g = *m.graph();
    return std::make_shared<SimpleValue>(
        g.insert(aten::__derive_index, {idx, start_, step_}, {}, loc));
  }
}

std::vector<SugaredValuePtr> IterableTree::get_base_iterables() {
  std::vector<SugaredValuePtr> base_iters{};

  for (SugaredValuePtr& sv : children_) {
    if (auto iv = std::dynamic_pointer_cast<IterableTree>(sv)) {
      std::vector<SugaredValuePtr> child_iters = iv->get_base_iterables();
      // merge child iters with the base_iters
      base_iters.insert(
          base_iters.end(),
          std::make_move_iterator(child_iters.begin()),
          std::make_move_iterator(child_iters.end()));

    } else {
      // IterableTree leaves, either SimpleValue or RangeValue
      base_iters.emplace_back(sv);
    }
  }
  return base_iters;
}

Value* IterableTree::len(const SourceRange& loc, GraphFunction& m) {
  // if it's a iterable tree, we get the base iterables that consists of
  // SimpleValue or RangeValue, and then calculate the minimum length of all the
  // base iterables to be max_trip_count_val
  TORCH_INTERNAL_ASSERT(!unroll_length_);
  Graph& g = *m.graph();
  std::vector<SugaredValuePtr> base_iters = get_base_iterables();
  std::vector<Value*> lengths;
  lengths.reserve(base_iters.size());

  for (const SugaredValuePtr& base_iter : base_iters) {
    lengths.emplace_back(base_iter->len(loc, m));
  }
  Node* list_node = g.insertNode(g.createList(IntType::get(), lengths));
  return g.insert(prim::min, {list_node->output()}, {}, loc);
}

SugaredValuePtr IterableTree::getitem(
    const SourceRange& loc,
    GraphFunction& m,
    Value* idx,
    TypePtr type_hint) {
  std::vector<SugaredValuePtr> child_items;
  child_items.reserve(children_.size());
  for (const SugaredValuePtr& child : children_) {
    child_items.emplace_back(child->getitem(loc, m, idx));
  }
  return std::make_shared<SugaredTupleValue>(child_items);
}

void IterableTree::addChild(
    const SourceRange& range,
    GraphFunction& m,
    const SugaredValuePtr& iter_value) {
  std::optional<int64_t> child_len = iter_value->staticLen();
  if (children_.empty()) {
    unroll_length_ = child_len;
  } else {
    if ((unroll_length_ && !child_len) || (child_len && !unroll_length_)) {
      throw ErrorReport(range)
          << "Can not iterate over a module list or tuple with a value "
             "that does not have a statically determinable length\n";
    }
    if (unroll_length_ && child_len) {
      // iterables run for the minimum length of all its leaves
      unroll_length_ = std::min(*child_len, *unroll_length_);
    } else {
      unroll_length_ = c10::nullopt;
    }
  }
  children_.push_back(iter_value);
}

std::shared_ptr<SugaredValue> MagicMethod::call(
    const SourceRange& loc,
    GraphFunction& m,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    size_t n_binders) {
  if (!args.empty()) {
    Value* self = args[0].value(*m.graph());
    if (auto class_ptr = self->type()->cast<ClassType>()) {
      return SimpleValue(self)
          .attr(loc, m, desugared_name_)
          ->call(loc, m, args.slice(1), kwargs, n_binders);
    }
  }
  TORCH_INTERNAL_ASSERT(base_value_);
  return base_value_->call(loc, m, args, kwargs, n_binders);
}

std::shared_ptr<SugaredValue> ClassValue::call(
    const SourceRange& loc,
    GraphFunction& m,
    // note: names for args will be 'argument 0', 'argument 1', etc..
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    size_t n_binders) {
  AT_ASSERT(n_binders <= 1);

  // Generate a new object of the right type, then call `__init__` on it
  auto& g = *m.graph();
  auto self = g.insertNode(g.createObject(type_))->output();
  self->node()->setSourceRange(loc);
  if (!type_->findMethod("__init__")) {
    throw ErrorReport(loc) << "Class " << type_->name()->name()
                           << " does not have an __init__ function defined";
  }

  // Call the init function
  MethodValue(self, "__init__").call(loc, m, args, kwargs, n_binders);

  return std::make_shared<SimpleValue>(self);
}

std::shared_ptr<SugaredValue> ClassValue::attr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) {
  // Allow import_source.cpp to resolve calls to a submodule's
  // hooks. Edge case because normally you wouldn't allow a module to
  // call functions of a submodule
  if (Function* hook = type_->findHook(field)) {
    return std::make_shared<FunctionValue>(hook);
  }

  if (field != "__new__") {
    throw ErrorReport(loc) << "Tried to lookup unknown attribute on class "
                           << type_->annotation_str();
  }
  return SpecialFormValue::create(prim::CreateObject);
}

std::shared_ptr<SugaredValue> NamedTupleConstructor::call(
    const SourceRange& loc,
    GraphFunction& m,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    size_t n_binders) {
  auto& g = *m.graph();

  auto schema = type_->schema();
  TORCH_INTERNAL_ASSERT(schema);
  auto qualname = type_->name();
  auto matched_schema = matchSchema(*schema, loc, g, args, kwargs);

  auto self =
      g.insertNode(
           g.createTuple(matched_schema.inputs, type_)->setSourceRange(loc))
          ->output();
  self->setType(type_);

  return std::make_shared<SimpleValue>(self);
}

std::shared_ptr<BuiltinFunction> BuiltinFunction::tryCreate(
    Symbol symbol,
    std::optional<NamedValue> self) {
  for (const std::shared_ptr<Operator>& op : getAllOperatorsFor(symbol)) {
    if (!self) {
      return std::make_shared<BuiltinFunction>(symbol, nullptr);
    }
    if (auto index = op->schema().argumentIndexWithName("self")) {
      std::unordered_map<std::string, TypePtr> type_env;
      TypePtr formal_type = op->schema().arguments().at(*index).type();
      const MatchTypeReturn matched =
          matchTypeVariables(formal_type, self->type(), type_env);
      if (!matched.success()) {
        continue;
      }
      const auto concrete_type = tryEvalTypeVariables(formal_type, type_env);
      if (!concrete_type || !self->type()->isSubtypeOf(*concrete_type)) {
        continue;
      }
      return std::make_shared<BuiltinFunction>(symbol, self);
    }
  }
  return nullptr;
}

std::shared_ptr<SugaredValue> SugaredEnumClass::attr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& field) {
  const auto& names_values = enum_type_->enumNamesValues();
  auto it = std::find_if(
      names_values.begin(),
      names_values.end(),
      [&field](const at::EnumNameValue& nv) { return nv.first == field; });
  if (it == names_values.end()) {
    throw ErrorReport(loc) << enum_type_->repr_str() << "'"
                           << " has no attribute '" << field << "'";
  }
  auto enum_holder = c10::make_intrusive<at::ivalue::EnumHolder>(
      enum_type_, it->first, it->second);
  return std::make_shared<SimpleValue>(
      m.graph()->insertConstant(IValue(enum_holder), loc));
}

SugaredValuePtr SugaredEnumClass::iter(
    const SourceRange& loc,
    GraphFunction& m) {
  const auto& names_values = enum_type_->enumNamesValues();
  auto enum_value_ivalues = c10::impl::GenericList(enum_type_);
  enum_value_ivalues.reserve(names_values.size());
  for (const auto& name_value : names_values) {
    auto enum_holder = c10::make_intrusive<at::ivalue::EnumHolder>(
        enum_type_, name_value.first, name_value.second);
    enum_value_ivalues.emplace_back(enum_holder);
  }

  auto enum_values_list_constant = std::make_shared<SimpleValue>(
      m.graph()->insertConstant(enum_value_ivalues, loc));
  return enum_values_list_constant;
}

} // namespace torch::jit
