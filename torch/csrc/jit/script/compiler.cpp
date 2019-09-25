#include <torch/csrc/jit/script/compiler.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/jit/hooks_for_testing.h>
#include <torch/csrc/jit/interpreter.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inline_forked_closures.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lift_closures.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/script/canonicalize_modified_loop.h>
#include <torch/csrc/jit/script/convert_to_ssa.h>
#include <torch/csrc/jit/script/parser.h>
#include <torch/csrc/jit/script/schema_matching.h>
#include <torch/csrc/jit/script/script_type_parser.h>

#include <torch/csrc/jit/constants.h>

#include <c10/util/Optional.h>

#include <atomic>
#include <climits>
#include <set>

namespace torch {
namespace jit {
namespace script {

using FunctionTable = std::unordered_map<std::string, Function&>;
using ValueTable = std::unordered_map<std::string, SugaredValuePtr>;
using TypeTable = std::unordered_map<std::string, TypePtr>;
using AttributeMap = std::unordered_map<std::string, Const>;
using ListAttributeMap = std::unordered_map<std::string, std::vector<Const>>;

struct Refinement {
  Refinement(std::string identifier, TypePtr type)
      : identifier_(std::move(identifier)), type_(type) {}
  const std::string& identifier() const {
    return identifier_;
  }
  TypePtr type() const {
    return type_;
  }

 private:
  std::string identifier_;
  TypePtr type_;
};

struct RefinementSet {
  // When a comparison like x is None is made, we associate type refinements
  // with its true value and its false value. If a boolean that has refinements
  // associated with it is used in a conditional of an if statememt, the true
  // and false refinements are inserted into the corresponding blocks
  using Refinements = std::vector<Refinement>;

  RefinementSet(Refinements true_refinements, Refinements false_refinements)
      : true_refinements_(std::move(true_refinements)),
        false_refinements_(std::move(false_refinements)) {}
  RefinementSet(Refinement single) : RefinementSet({std::move(single)}, {}) {}
  RefinementSet(Refinement single_true, Refinement single_false)
      : RefinementSet(
            Refinements({std::move(single_true)}),
            Refinements({std::move(single_false)})) {}
  RefinementSet() {} // empty
  RefinementSet And(const RefinementSet& rhs) const {
    // if the result of an AND is true, both a & b had to be true,
    // so we take the union of a.true_refinements and b.true_refinements.
    // if the result is false, either a or b could have been false,
    // so we take their intersection.
    return RefinementSet(
        unionSet(true_refinements_, rhs.true_refinements_),
        intersectSet(false_refinements_, rhs.false_refinements_));
  }
  RefinementSet Or(const RefinementSet& rhs) const {
    // if the result of an OR is true, either a & b could have been true,
    // so we take the intersection of a.true_refinements & b.true_refinements.
    // if the result is false, both a and b had to be false,
    // so we take their union.
    return RefinementSet(
        intersectSet(true_refinements_, rhs.true_refinements_),
        unionSet(false_refinements_, rhs.false_refinements_));
  }

  RefinementSet Not() const {
    return RefinementSet(false_refinements_, true_refinements_);
  }
  const std::vector<Refinement> activeRefinements() const {
    return true_refinements_;
  }

 private:
  static bool sameVar(const Refinement& a, const Refinement& b) {
    return a.identifier() == b.identifier();
  }
  static Refinements unionSet(const Refinements& a, const Refinements& b) {
    Refinements result = a;
    for (const Refinement& r : b) {
      auto it =
          std::find_if(result.begin(), result.end(), [&](const Refinement& e) {
            return e.identifier() == r.identifier();
          });
      if (it == result.end()) {
        result.push_back(r);
      } else if (*it->type() != *r.type()) {
        // we only keep refinements when they exactly match one
        // refinement type, for instance, we do not attempt to refine:
        // isinstance(x, float) and isinstance(x, int)
        result.erase(it);
      }
    }
    return result;
  }
  static Refinements intersectSet(const Refinements& a, const Refinements& b) {
    Refinements result;
    for (const Refinement& r : a) {
      auto it = std::find_if(b.begin(), b.end(), [&](const Refinement& e) {
        return e.identifier() == r.identifier();
      });
      if (it != b.end() && r.type() == it->type()) {
        result.push_back(r);
      }
    }
    return result;
  }

  Refinements true_refinements_;
  Refinements false_refinements_;
};

struct CondValue {
  CondValue(
      Value* value,
      RefinementSet refinements,
      c10::optional<bool> static_if)
      : value_(value),
        refinements_(std::move(refinements)),
        static_if_(static_if) {}
  CondValue(
      Graph& g,
      const SourceRange& loc,
      bool static_value,
      RefinementSet refinements)
      : value_(g.insertConstant(static_value, loc)),
        refinements_(std::move(refinements)),
        static_if_(static_value) {}
  Value* value() const {
    return value_;
  }
  const RefinementSet& refinements() const {
    return refinements_;
  }
  c10::optional<bool> staticIf() const {
    return static_if_;
  }

 private:
  Value* value_;
  RefinementSet refinements_;
  c10::optional<bool>
      static_if_; // certain expression cause us to emit a static if statement
                  // this value is present if this is the case.
                  // this is not equivalent to value_ being a constant
                  // it is possible for value_ to be constant but for
                  // the expression that produced it to not trigger the
                  // static if behavior. e.g. use of a variable assigned
                  // to a constant
};

enum NoneStatus { ALWAYS, MAYBE, NEVER };
NoneStatus canBeNone(Value* v) {
  if (v->node()->mustBeNone()) {
    return ALWAYS;
  }
  if (v->type()->kind() == OptionalType::Kind) {
    return MAYBE;
  }
  return NEVER;
}

static Value* asSimple(const SugaredValuePtr& value) {
  if (SimpleValue* sv = dynamic_cast<SimpleValue*>(value.get())) {
    return sv->getValue();
  }
  return nullptr;
}

static std::shared_ptr<MagicMethod> makeMagic(
    const std::string& name,
    SugaredValuePtr base) {
  return std::make_shared<MagicMethod>(name, base);
}

// Auxiliary data structure for desugaring variable binding into our always
// explicitly scoped language as we descend down nested control structures in
// the frontend (which themselves don't introduce scopes)
//
// The Environment keeps track of two tables, one for values which are not first
// class and a type table for values which are. When a first class value
// is set in the environment, we emit a prim::Store which sets the
// name of the variable to approriate type, and when a first-class value is
// referenced we emit a prim::Load that generates a value of the appropriate
// type.
//
// a = 1
// print(a)
// becomes:
// = prim::Store[name="a"](%a.1)
// %a : int = prim::Load[name="a"]()
// prim::Print(%a)

struct Environment {
  Environment(
      Function& method,
      ResolverPtr resolver,
      Block* b,
      std::shared_ptr<Environment> next = nullptr)
      : method(method),
        resolver(std::move(resolver)),
        b(b),
        next(std::move(next)) {}

  Function& method;
  ResolverPtr resolver;
  std::unordered_map<std::string, std::function<std::string()>> error_messages;
  Block* b;

  std::shared_ptr<Environment> next;

  // set type error in the lowest environment. if the variable is used after an
  // error has been set, then we will use the more informative error message
  void setVariableTypeError(
      const std::string& name,
      std::function<std::string()> msg) {
    auto runner = this;
    while (runner->next) {
      runner = runner->next.get();
    }
    runner->error_messages[name] = msg;
  }

  // see if type error has been set for a variable
  c10::optional<std::string> findVariableTypeError(const std::string& name) {
    auto runner = this;
    while (runner->next) {
      runner = runner->next.get();
    }
    auto msg = runner->error_messages.find(name);
    if (msg != runner->error_messages.end()) {
      return msg->second();
    } else {
      return c10::nullopt;
    }
  }

  SugaredValuePtr insertLoad(const std::string& name, const TypePtr& type) {
    auto g = b->owningGraph();
    auto load = g->insertNode(g->createLoad(name, type));
    if (meaningfulName(name)) {
      load->output()->setDebugName(name);
    }
    return std::make_shared<SimpleValue>(load->output());
  }

  // note: type is not always the same as v->type(), e.g.
  // type: Optional[Tensor]
  // v->type(): Tensor
  void insertStore(
      const std::string& name,
      const SourceRange& loc,
      Value* v,
      TypePtr type) {
    auto g = b->owningGraph();
    g->insertNode(g->createStore(name, v))->setSourceRange(loc);
    type_table[name] = type;
  }

  SugaredValuePtr findInThisFrame(const std::string& name) {
    auto it = value_table.find(name);
    if (it != value_table.end()) {
      return it->second;
    }
    auto it2 = type_table.find(name);
    if (it2 != type_table.end()) {
      return insertLoad(name, it2->second);
    }
    return nullptr;
  }

  SugaredValuePtr findInParentFrame(const std::string& name) {
    return next ? next->findInAnyFrame(name) : nullptr;
  }

  void setType(const std::string& name, TypePtr type) {
    type_table[name] = std::move(type);
  }

  SugaredValuePtr findInAnyFrame(const std::string& name) {
    for (auto runner = this; runner; runner = runner->next.get()) {
      if (auto r = runner->findInThisFrame(name)) {
        return r;
      }
    }
    return nullptr;
  }

  Block* block() {
    return b;
  }

  void setVar(const SourceRange& loc, const std::string& name, Value* value) {
    setSugaredVar(
        loc,
        name,
        std::make_shared<SimpleValue>(value),
        /*annotated_type=*/nullptr);
  }

  void setSugaredVar(
      const SourceRange& loc,
      const std::string& name,
      SugaredValuePtr value,
      TypePtr annotated_type) {
    Value* as_simple_value = asSimple(value);
    if (as_simple_value && !as_simple_value->hasDebugName() &&
        meaningfulName(name) &&
        // note: if the value wasn't defined in this block, we might be giving a
        // name only used inside this block to a value outside of this. this is
        // not normally helpful for debugging and causes import/export jitter.
        as_simple_value->node()->owningBlock() == block()) {
      as_simple_value->setDebugName(name);
    }
    // prevent re-assignment involving any sugared values
    // any reassignment like:
    // a = ...
    // while ...
    //   a = ..
    // requires 'a' to be first-class in the graph since its value depends on
    // control flow
    if (auto parent = findInParentFrame(name)) {
      if (annotated_type) {
        throw ErrorReport(loc)
            << "Attempting to declare and annotate the type of variable '"
            << name << "' but it is already defined in an outer block";
      }
      if (!as_simple_value) {
        throw ErrorReport(loc)
            << "Cannot re-assign '" << name << "' to a value of type "
            << value->kind() << " because " << name
            << " is not a first-class value.  Only reassignments to first-class values are allowed";
      }
      Value* simple_parent = asSimple(parent);
      if (!simple_parent) {
        throw ErrorReport(loc)
            << "Cannot re-assign '" << name << "' because it has type "
            << value->kind() << " and " << name
            << " is not a first-class value.  Only reassignments to first-class values are allowed";
      }

      auto parent_type = unshapedType(simple_parent->type());
      as_simple_value = tryConvertToType(
          loc,
          *b->owningGraph(),
          parent_type,
          as_simple_value,
          /*allow_conversions=*/true);
      std::stringstream why_not;
      if (!as_simple_value->type()->isSubtypeOfExt(parent_type, &why_not)) {
        auto error = ErrorReport(loc);
        error << "Variable '" << name << "' previously has type "
              << simple_parent->type()->python_str()
              << " but is now being assigned to a value of type "
              << as_simple_value->type()->python_str();

        // Special-cased error msg if we're trying to assign to a tensor list.
        if (simple_parent->type()->kind() == TypeKind::ListType &&
            as_simple_value->type()->kind() == TypeKind::ListType) {
          error << "\nEmpty lists default to List[Tensor]. Add a variable "
                   "annotation to the assignment to create an empty list "
                   "of another type (torch.jit.annotate(List[T, []]) where T "
                   "is the type of elements in the list for Python 2)";
        }
        error << "\n" << why_not.str();
        throw error;
      }
    }
    if (as_simple_value) {
      if (!annotated_type) {
        annotated_type = as_simple_value->type();
      }
      if (!as_simple_value->type()->isSubtypeOf(annotated_type)) {
        throw ErrorReport(loc)
            << "Variable '" << name << "' is annotated with type "
            << annotated_type->python_str()
            << " but is being assigned to a value of type "
            << as_simple_value->type()->python_str();
      }
      insertStore(name, loc, std::move(as_simple_value), annotated_type);
    } else {
      value_table[name] = std::move(value);
    }
  }

  SugaredValuePtr getSugaredVar(const Ident& ident, bool required = true) {
    return getSugaredVar(ident.name(), ident.range());
  }
  Value* getVar(const Ident& ident) {
    return getSugaredVar(ident)->asValue(ident.range(), method);
  }

  SugaredValuePtr getSugaredVar(
      const std::string& ident,
      const SourceRange& range,
      bool required = true) {
    auto retval = findInAnyFrame(ident);

    if (!retval) {
      static std::unordered_map<std::string, SugaredValuePtr> globals = {
          {"print", std::make_shared<PrintValue>()},
          {"tuple", std::make_shared<TupleCallValue>()},
          {"float",
           makeMagic(
               "__float__",
               std::make_shared<CastValue>(FloatType::get(), aten::Float))},
          {"int",
           makeMagic(
               "__int__",
               std::make_shared<CastValue>(IntType::get(), aten::Int))},
          {"bool",
           makeMagic(
               "__bool__",
               std::make_shared<CastValue>(BoolType::get(), aten::Bool))},
          {"str",
           makeMagic(
               "__str__",
               std::make_shared<CastValue>(StringType::get(), aten::str))},
          {"getattr", std::make_shared<GetAttrValue>()},
          {"isinstance", std::make_shared<IsInstanceValue>()},
          // todo(zach): remove when we can correctly export torch.full via ONNX
          // or we have implicit conversion that can convert numbers to tensors
          {"_to_tensor",
           std::make_shared<CastValue>(TensorType::get(), prim::NumToTensor)},
          {"len",
           makeMagic(
               "__len__",
               std::make_shared<BuiltinFunction>(aten::len, at::nullopt))},
          {"hex",
           makeMagic(
               "__hex__",
               std::make_shared<BuiltinFunction>(aten::hex, at::nullopt))},
          {"oct",
           makeMagic(
               "__oct__",
               std::make_shared<BuiltinFunction>(aten::oct, at::nullopt))},
          {"round",
           makeMagic(
               "__round__",
               std::make_shared<BuiltinFunction>(aten::round, at::nullopt))},
          {"hash", std::make_shared<BuiltinFunction>(aten::hash, at::nullopt)},
          {"min", std::make_shared<BuiltinFunction>(prim::min, at::nullopt)},
          {"max", std::make_shared<BuiltinFunction>(prim::max, at::nullopt)},
          {"abs", std::make_shared<BuiltinFunction>(prim::abs, at::nullopt)},
          {"all", std::make_shared<BuiltinFunction>(aten::all, at::nullopt)},
          {"divmod",
           std::make_shared<BuiltinFunction>(aten::divmod, at::nullopt)},
          {"list", std::make_shared<BuiltinFunction>(aten::list, at::nullopt)},
          {"ord", std::make_shared<BuiltinFunction>(aten::ord, at::nullopt)},
          {"chr", std::make_shared<BuiltinFunction>(aten::chr, at::nullopt)},
          {"bin", std::make_shared<BuiltinFunction>(aten::bin, at::nullopt)},
          {"range", std::make_shared<IterableValue>(prim::range)},
          {"zip", std::make_shared<IterableValue>(prim::zip)},
          {"enumerate", std::make_shared<IterableValue>(prim::enumerate)},
          {"rangelist",
           std::make_shared<BuiltinFunction>(prim::rangelist, at::nullopt)},
          {"sorted",
           std::make_shared<BuiltinFunction>(aten::sorted, at::nullopt)},
      };
      auto it = globals.find(ident);
      if (it != globals.end()) {
        retval = it->second;
      }
    }

    if (!retval) {
      if (auto type = resolver->resolveType(ident, range)) {
        if (auto class_type = type->cast<ClassType>()) {
          retval = std::make_shared<script::ClassValue>(class_type);
        } else if (auto tuple_type = type->cast<TupleType>()) {
          retval = std::make_shared<script::NamedTupleConstructor>(tuple_type);
        }
      }
    }

    if (!retval) {
      retval = resolver->resolveValue(ident, method, range);
    }

    if (!retval && required) {
      // check if this value was not emitted in an if statement because of a
      // type mismatch. if it was, then we print a more informative error msg
      if (auto msg = findVariableTypeError(ident)) {
        throw ErrorReport(range) << *msg << "and was used here";
      }
      throw ErrorReport(range) << "undefined value " << ident;
    }
    return retval;
  }

  Value* getVar(const std::string& ident, const SourceRange& range) {
    return getSugaredVar(ident, range)->asValue(range, method);
  }

  std::vector<std::string> definedVariables() {
    std::vector<std::string> result;
    for (auto& kv : type_table) {
      result.push_back(kv.first);
    }
    return result;
  }

 private:
  TypeTable type_table;
  ValueTable value_table;
};

template <class T>
static Value* materializeConstant(
    T val,
    Graph& graph,
    const SourceRange& r,
    std::unordered_map<T, Value*>& map) {
  auto existing_constant = map.find(val);
  if (existing_constant != map.end()) {
    return existing_constant->second;
  }

  WithInsertPoint guard(graph.block()->nodes().front());
  auto new_constant = graph.insertConstant(val, r);
  map[val] = new_constant;

  return new_constant;
}

inline bool isSupportedListElementType(const TypePtr& type) {
  return type->isSubtypeOf(TensorType::get()) ||
      type->isSubtypeOf(NumberType::get());
}

// Information for each def being emitted.
// Defs can be nested to support closures so we need a stack of this information
// Currently records information about the functions return type.
struct DefContext {
  TypePtr declared_return_type_; // nullptr if not annotated
  TypePtr merged_return_type_; // nullptr if a Return has not been seen yet
};

struct to_ir {
  to_ir(
      const Def& def,
      ResolverPtr resolver_,
      const Self* self,
      Function& method) // method being constructed
      : method(method),
        graph(method.graph()),
        resolver(std::move(resolver_)),
        typeParser_(resolver),
        environment_stack(nullptr) {
    AT_ASSERT(resolver);
    pushFrame(graph->block(), /*starts_def=*/true);

    // Type annotations exclude explicitly typing the "self" parameter, so in
    // the case that this is a method with self we expect one fewer parameter
    // annotation than the number of parameters this Def takes.
    if (self && def.decl().params().size() == 0) {
      throw ErrorReport(def.decl().params().range())
          << "methods must have a self argument";
    }
    method.setSchema(emitDef(def, self, graph->block()));
    runCleanupPasses(graph);
  }

 private:
  Function& method;
  std::shared_ptr<Graph> graph;
  ResolverPtr resolver;
  std::unordered_map<int64_t, Value*> integral_constants;
  std::unordered_map<double, Value*> fp_constants;
  std::unordered_set<Block*> exit_blocks;
  ScriptTypeParser typeParser_;

  // Singly-linked list of environments. This top element contains a member
  // `next` that points to the most immediate enclosing scope's value.
  std::shared_ptr<Environment> environment_stack;
  std::vector<DefContext> def_stack_;
  size_t temp_name_count_ = 0;
  std::string createTempName(const std::string& prefix) {
    return prefix + std::to_string(temp_name_count_++);
  }

  void pushFrame(Block* b, bool starts_def = false) {
    if (starts_def) {
      def_stack_.emplace_back();
    }
    environment_stack =
        std::make_shared<Environment>(method, resolver, b, environment_stack);
  }
  std::shared_ptr<Environment> popFrame(bool ends_def = false) {
    auto old_frame = environment_stack;
    environment_stack = environment_stack->next;
    if (ends_def) {
      def_stack_.pop_back();
    }
    return old_frame;
  }

  // If the graph might not return, add an implicit None return at the end
  void handleMaybeNoReturn(const Def& def, Block* block) {
    auto decl_ret = def_stack_.back().declared_return_type_;
    if (exit_blocks.count(block) == 0) {
      auto decl_ret = def_stack_.back().declared_return_type_;
      if (decl_ret && decl_ret != NoneType::get()) {
        throw ErrorReport(def.range())
            << "Function was not annotated as having type None, but does not "
            << "return along all paths";
      }
      WithInsertPoint b(*block->nodes().end());
      emitReturn(Return::create(
          def.range(), Expr(Compound::create(TK_NONE, def.range(), {}))));
    } else {
      // if we haven't seen any return statements, but the graph block exits
      // (the funciton always throws) then we accept the declared return type if
      // it exists or set it to none
      if (def_stack_.back().merged_return_type_ == nullptr) {
        def_stack_.back().merged_return_type_ =
            decl_ret != nullptr ? decl_ret : NoneType::get();
      }
    }
  }

  FunctionSchema emitDef(const Def& def, const Self* self, Block* block) {
    auto schema = typeParser_.parseSchemaFromDef(def, bool(self));
    // TODO need guards on init returning none
    if (schema.returns().size() == 1) {
      def_stack_.back().declared_return_type_ = schema.returns().at(0).type();
    }
    std::vector<Argument> arguments =
        emitFormalArguments(def, self, schema, block);

    // body
    auto stmts_list = def.statements();
    emitStatements(stmts_list.begin(), stmts_list.end());
    handleMaybeNoReturn(def, block);
    std::vector<Argument> returns = {emitOutput(def.range(), schema, block)};
    return {def.name().name(), "", std::move(arguments), std::move(returns)};
  }

  // see [setstate type]
  static TypePtr getTypeForSetStateArg(const Self* self) {
    TORCH_CHECK(self, "Expected __setstate__ to have a `self` argument");
    self->getClassType()->getMethod("__getstate__")->ensure_defined();
    return self->getClassType()
        ->getMethod("__getstate__")
        ->getSchema()
        .returns()
        .at(0)
        .type();
  }

  // see [setstate type]
  static bool shouldDeriveSetStateType(
      const Def& def,
      const FunctionSchema& schema) {
    const bool noTypeAnnotations = std::all_of(
        schema.arguments().begin(),
        schema.arguments().end(),
        [](const Argument& arg) { return arg.is_inferred_type(); });

    bool shouldInfer = def.name().name() == "__setstate__" && noTypeAnnotations;
    if (!shouldInfer) {
      return false;
    }

    // Do some additional basic validation that the __setstate__ func is
    // well-formed
    TORCH_INTERNAL_ASSERT(def.name().name() == "__setstate__");
    const auto numDeclParams = def.decl().params().size();
    TORCH_CHECK(
        numDeclParams,
        "Expected 2 arguments for __setstate__, got: ",
        numDeclParams);
    return true;
  }

  std::vector<Argument> emitFormalArguments(
      const Def& def,
      const Self* self,
      const FunctionSchema& schema,
      Block* block) {
    std::vector<Argument> arguments; // for schema
    // inputs
    auto it = def.decl().params().begin();
    auto end = def.decl().params().end();
    auto expected_annotation_size = def.decl().params().size();
    if (self) {
      expected_annotation_size--;
    }
    if (schema.arguments().size() != expected_annotation_size) {
      throw ErrorReport(def.decl().params().range())
          << "Number of type annotations for"
          << " function parameters (" << schema.arguments().size() << ")"
          << " does not match the number of parameters on the function ("
          << expected_annotation_size << ")!";
    }

    if (self) {
      AT_ASSERT(it != end);
      const auto& name = (*it).ident().name();
      Value* new_input = block->addInput()->setDebugName(name);
      environment_stack->setSugaredVar(
          (*it).ident().range(),
          name,
          self->makeSugared(new_input),
          /*annotated_type=*/nullptr);
      arguments.emplace_back(name, new_input->type());
      ++it;
    }

    // [setstate type]
    // __setstate__ is special, because if the user leaves it un-annotated we
    // will derive the type for `state` from the output type of __getstate__.
    // This is necessary so that we can allow submodules to appear in `state`.
    bool shouldDeriveType = shouldDeriveSetStateType(def, schema);
    size_t arg_annotation_idx = 0;
    for (; it != end; ++it) {
      auto& name = (*it).ident().name();
      // Add the input to the graph
      Value* new_input = block->addInput();
      if (meaningfulName(name)) {
        new_input->setDebugName(name);
      }
      // Record the type for the schema and set the Type on the Value*
      auto arg = schema.arguments().at(arg_annotation_idx++);
      if (shouldDeriveType) {
        TORCH_INTERNAL_ASSERT(schema.arguments().size() == 1);
        const auto& inferredStateType = getTypeForSetStateArg(self);
        arg = arg.cloneWithType(inferredStateType);
      }

      arguments.push_back(arg);
      new_input->setType(arguments.back().type());

      // NB: set type of new_input before setVar call so the Store is
      // typed appropriately
      environment_stack->setVar((*it).ident().range(), name, new_input);
    }
    return arguments;
  }

  Argument emitOutput(
      const SourceRange& range,
      const FunctionSchema& schema,
      Block* block) {
    // handleMaybeNoReturn ensures that merged_return_type_ is always set
    auto ret_type = def_stack_.back().merged_return_type_;
    TORCH_INTERNAL_ASSERT(ret_type);

    // in the ConvertToSSA pass, prim::ReturnStmts are lowered so that the
    // correct return value is set. Until then, we have a correctly-typed
    // placeholder return value. This is needed so that closures & graphs
    // are correctly typed.
    auto placeholder_return =
        graph->insertNode(graph->createUninitialized(ret_type))->output();
    block->registerOutput(placeholder_return);
    return Argument("", def_stack_.back().merged_return_type_);
  }

  void emitStatements(const List<Stmt>& statements) {
    return emitStatements(statements.begin(), statements.end());
  }

  // XXX - right now closures are used _only_ for defining gradients internally
  // There are several unfinished aspects that make them unusable generally
  // 1. We do not have a type, ivalue, operator to represent prim::Function, so
  // closure_node has type None
  // 2. There is no export logic for it yet, so it cannot be
  // exported/python_printed
  // 3. There is nothing preventing the assignment of already existing variables
  // inside the closures
  //    the changes to those variables will just get forgotten.
  // 4. There is no parsing support in frontend.py, this is intentional since it
  //    prevents people from accidentally using this feature.
  std::shared_ptr<ClosureValue> emitClosure(
      const std::function<void(Block*)>& emit_body) {
    Node* closure_node = graph->insertNode(graph->create(prim::Function, 1));
    // it is not a real thing yet, so just say the type is None
    closure_node->output()->setType(NoneType::get());
    Block* block = closure_node->addBlock();
    {
      WithInsertPoint guard(block);
      pushFrame(block, /*starts_def=*/true);
      emit_body(block);
      popFrame(/*ends_def=*/true);
    }
    return std::make_shared<ClosureValue>(closure_node->output());
  }

  void emitClosure(const Def& def) {
    // invoked once the closure block is set as the enviroment
    auto emit_body = [&](Block* closure_block) {
      emitDef(
          def,
          nullptr,
          closure_block); // ignore schema return, we just wont use it for now
                          // since we never create a Method for the closure
    };
    auto closure_value = emitClosure(emit_body);
    environment_stack->setSugaredVar(
        def.name().range(),
        def.name().name(),
        closure_value,
        /*annotated_type=*/nullptr);
  }

  void emitBreak(const Break& stmt) {
    auto break_node =
        graph->create(prim::BreakStmt, {}, 0)->setSourceRange(stmt.range());
    graph->insertNode(break_node);
  }

  void emitContinue(const Continue& stmt) {
    auto continue_node =
        graph->create(prim::ContinueStmt, {}, 0)->setSourceRange(stmt.range());
    graph->insertNode(continue_node);
  }

  void emitReturn(const Return& stmt) {
    Value* result = emitExpr(stmt.expr());
    TypePtr result_type = def_stack_.back().declared_return_type_;
    // result type is annotated, every return must convert to that type
    if (result_type) {
      // this guard skips implicit conversion from None -> Tensor for the return
      // type. otherwise forgetting a return a function returning a tensor will
      // cause a None to be converted to a tensor.
      if (!(result_type->isSubtypeOf(TensorType::get()) &&
            result->type()->isSubtypeOf(NoneType::get()))) {
        result = tryConvertToType(
            stmt.range(),
            *graph,
            result_type,
            result,
            /*allow_conversions=*/true);
      }

      if (!result->type()->isSubtypeOf(result_type)) {
        throw ErrorReport(stmt.range())
            << "Return value was annotated as having type "
            << result_type->python_str() << " but is actually of type "
            << result->type()->python_str();
      }
    } else {
      result_type = def_stack_.back().merged_return_type_;
      if (!result_type) {
        result_type = result->type();
      }
      auto merged_result_type = unifyTypes(result_type, result->type());
      if (!merged_result_type) {
        throw ErrorReport(stmt.range())
            << "Previous return statement returned a value of type "
            << result_type->python_str()
            << " but this return statement returns a value of type "
            << result->type()->python_str();
      }
      result_type = merged_result_type.value();
    }
    AT_ASSERT(result_type);
    def_stack_.back().merged_return_type_ = result_type;
    graph->insertNode(graph->create(prim::ReturnStmt, {result}, 0));
    exit_blocks.insert(environment_stack->block());
  }

  void emitStatements(
      List<Stmt>::const_iterator begin,
      List<Stmt>::const_iterator end) {
    for (; begin != end; ++begin) {
      auto stmt = *begin;
      ErrorReport::CallStack::update_pending_range(stmt.range());
      switch (stmt.kind()) {
        case TK_IF:
          emitIf(If(stmt));
          break;
        case TK_WHILE:
          emitWhile(While(stmt));
          break;
        case TK_FOR:
          emitFor(For(stmt));
          break;
        case TK_ASSIGN:
          emitAssignment(Assign(stmt));
          break;
        case TK_AUG_ASSIGN:
          emitAugAssignment(AugAssign(stmt));
          break;
        case TK_EXPR_STMT: {
          auto expr = ExprStmt(stmt).expr();
          emitSugaredExpr(expr, 0);
        } break;
        case TK_RAISE:
          emitRaise(Raise(stmt).range());
          break;
        case TK_ASSERT:
          emitAssert(Assert(stmt));
          break;
        case TK_RETURN: {
          emitReturn(Return(stmt));
        } break;
        case TK_CONTINUE: {
          emitContinue(Continue(stmt));
        } break;
        case TK_BREAK: {
          emitBreak(Break(stmt));
        } break;
        case TK_PASS:
          // Emit nothing for pass
          break;
        case TK_DEF:
          emitClosure(Def(stmt));
          break;
        default:
          throw ErrorReport(stmt)
              << "Unrecognized statement kind " << kindToString(stmt.kind());
      }
    }
  }

  RefinementSet findIsNoneRefinements(
      Expr lhs,
      Value* lhs_value,
      Expr rhs,
      Value* rhs_value,
      int tok) {
    if (rhs.kind() != TK_NONE && lhs.kind() == TK_NONE) {
      // make 'None is var' into 'var is None'
      return findIsNoneRefinements(rhs, rhs_value, lhs, lhs_value, tok);
    }
    if (rhs.kind() != TK_NONE || lhs.kind() != TK_VAR) {
      return {};
    }
    // statement must be var {is, is not} None
    auto name = Var(lhs).name().name();
    // XXX - while it should in theory be possible to specialize
    // the `x is None` to know x has type NoneType, we have previously not
    // done this. Unfortunately, doing this will make the type None
    // propagate further in all loaded models. The handling of
    // unwrap_optional will fail in these cases since export did
    // not expect that the input would be none and an unannotated None.
    // cannot be passed to unwrapoptional To enable this,
    // we need to (1) implement a real casting operator
    // annotated(T, X) that stays in the graph and does the cast
    // and (2) only enable this OPTIONAL_NONE when loading newer
    // graphs because it is incompatible with older graphs.
    // Refinement none(name, RefinementKind::OPTIONAL_NONE);
    if (auto optional_type = lhs_value->type()->cast<OptionalType>()) {
      Refinement present(name, optional_type->getElementType());
      if (tok == TK_IS) {
        return RefinementSet({}, {present});
      } else { // TK_ISNOT
        return RefinementSet({present}, {});
      }
    }
    return RefinementSet();
  }

  CondValue emitCondExpr(const Expr& expr) {
    switch (expr.kind()) {
      case TK_AND:
      case TK_OR: {
        auto binop = BinOp(expr);
        return emitShortCircuitLogical(
            binop.range(), binop.lhs(), binop.rhs(), expr.kind() == TK_OR);
      }
      case TK_NOT: {
        CondValue v = emitCondExpr(Expr(expr.tree()->trees()[0]));
        Value* result = emitBuiltinCall(
            expr.range(),
            *graph,
            aten::__not__,
            c10::nullopt,
            {v.value()},
            {},
            /*required=*/true);
        c10::optional<bool> static_if;
        if (v.staticIf()) {
          static_if = !*v.staticIf();
        }
        return CondValue(result, v.refinements().Not(), static_if);
      } break;
      case TK_IS:
      case TK_ISNOT: {
        // meta programming on AST for is/is not cases and emit branches base on
        auto cond_op = BinOp(expr);
        Value* lhs_val = emitExpr(cond_op.lhs());
        Value* rhs_val = emitExpr(cond_op.rhs());

        auto lhs_none = canBeNone(lhs_val);
        auto rhs_none = canBeNone(rhs_val);

        // Dispatch logic (A: ALWAYS, N: NEVER, M: MAYBE):
        //
        // AA, -> statically IS always holds, IS_NOT never holds
        // AN , NA-> statically IS_NOT always holds, IS never holds
        // MA, MM, MN, NM, NN, AM -> cannot prove anything statically
        bool its_is = expr.kind() == TK_IS;
        if (lhs_none == ALWAYS && rhs_none == ALWAYS) {
          return CondValue(*graph, expr.range(), its_is, {});
        } else if (
            (lhs_none == ALWAYS && rhs_none == NEVER) ||
            (lhs_none == NEVER && rhs_none == ALWAYS)) {
          // lhs_val/rhs_val with A/M: only emit never_none_branch
          return CondValue(*graph, expr.range(), !its_is, {});
        } else {
          auto kind = getNodeKind(expr.kind(), expr.get()->trees().size());
          Value* cond_value = emitBuiltinCall(
              expr.get()->range(),
              *method.graph(),
              kind,
              c10::nullopt,
              {lhs_val, rhs_val},
              {},
              /*required=*/true);
          auto refinements = RefinementSet(findIsNoneRefinements(
              cond_op.lhs(), lhs_val, cond_op.rhs(), rhs_val, expr.kind()));
          return CondValue(cond_value, refinements, c10::nullopt);
        }
      } break;
      default: {
        if (expr.kind() == TK_APPLY) {
          auto apply = Apply(expr);
          auto callee = Apply(expr).callee();
          if (callee.kind() == TK_VAR &&
              Var(callee).name().name() == "isinstance") {
            checkApplyNumInputs(apply, 2);
            return emitIsInstance(apply.inputs()[0], apply.inputs()[1]);
          }
        }
        return CondValue(
            emitToBool(emitExpr(expr)), RefinementSet({}), c10::nullopt);
      } break;
    }
  }

  std::shared_ptr<Environment> emitSingleIfBranch(
      Block* b,
      const List<Stmt>& branch,
      const RefinementSet& refinements) {
    pushFrame(b);
    WithInsertPoint guard(b);
    insertRefinements(branch.range(), refinements);
    emitStatements(branch);
    return popFrame();
  }

  Node* create(Symbol kind, const SourceRange& loc, size_t n_outputs) {
    return graph->create(kind, n_outputs)->setSourceRange(loc);
  }

  Value* emitTernaryIf(const TernaryIf& expr) {
    CondValue cond_value = emitCondExpr(expr.cond());
    auto true_expr = [&] { return emitExpr(expr.true_expr()); };
    auto false_expr = [&] { return emitExpr(expr.false_expr()); };
    return emitIfExpr(expr.range(), cond_value, true_expr, false_expr);
  }

  // emit a single expr from the loop comprehension so that we can correctly
  // type the list we create, then remove the nodes we emitted
  TypePtr getListCompType(
      const ListComp& lc,
      const ListTypePtr& input_list_type) {
    auto b = graph->insertNode(graph->create(prim::Loop))->addBlock();
    pushFrame(b);
    WithInsertPoint guard(b);
    auto li_elem = graph->insertNode(
        graph->createUninitialized(input_list_type->getElementType()));
    emitExprsAssign(
        List<Expr>::create(lc.range(), {lc.target()}),
        {std::make_shared<SimpleValue>(li_elem->output())},
        lc.range(),
        /*n_binders*/ 1);
    auto ret_type = emitExpr(lc.elt())->type();
    popFrame();
    b->owningNode()->destroy();
    return ret_type;
  }

  Value* emitListComprehension(const ListComp& lc) {
    const auto tmp_name = createTempName("$list_acc");
    const auto list_value = emitExpr(lc.iter());
    if (list_value->type()->kind() != TypeKind::ListType) {
      // TODO: constraining iterators to be simple lists for now
      // as it makes easy to get list's element type.
      throw ErrorReport(lc.range())
          << "iterator expression is expected to be a list";
    }

    // given `[x*2 for x in my_list]` this generates the following AST:
    // __list_acc = []
    // for x in my_list:
    //  __list_acc.append(x*2)
    const auto n = graph->insertNode(graph->createList(
        getListCompType(lc, list_value->type()->expect<ListType>()),
        at::ArrayRef<Value*>{}));
    environment_stack->setVar(lc.range(), tmp_name, n->output());
    const auto tmp_list_ident = Ident::create(lc.range(), tmp_name);
    const auto tmp_list_var = Var::create(lc.range(), tmp_list_ident);
    const auto append_ident = Ident::create(lc.range(), "append");
    const auto dot_op = Select::create(lc.range(), tmp_list_var, append_ident);
    const auto append_args_list = List<Expr>::create(lc.range(), {lc.elt()});
    const auto append_attrs = List<Attribute>::create(lc.range(), {});
    const auto apply_append =
        Apply::create(lc.range(), dot_op, append_args_list, append_attrs);
    const auto expr_stmt = ExprStmt::create(lc.range(), apply_append);
    const auto stmt_list = List<Stmt>::create(lc.range(), {expr_stmt});
    const auto iters_list = List<Expr>::create(lc.range(), {lc.iter()});
    const auto targets_list = List<Expr>::create(lc.range(), {lc.target()});
    const auto for_loop =
        For::create(lc.range(), targets_list, iters_list, stmt_list);
    emitFor(for_loop);
    return n->output();
  }

  // Insert subtyping refinements
  void insertRefinements(const SourceRange& loc, const RefinementSet& ref) {
    for (const Refinement& r : ref.activeRefinements()) {
      Value* v = environment_stack->getVar(r.identifier(), loc);
      Value* output = graph->insert(prim::unchecked_unwrap_optional, {v});
      environment_stack->setVar(loc, r.identifier(), output);
    }
  }

  CondValue emitShortCircuitLogical(
      const SourceRange& loc,
      const Expr& first_expr,
      const Expr& second_expr,
      bool is_or) {
    CondValue lhs = emitCondExpr(first_expr);
    // if the continue expr in the short circuit is not evaluated,
    // than the const expression is False if the short circuit
    // is an `and` and True if the short circuit is an `or`.
    // `False and expr` -> False, `True or expr` -> True
    //
    // inserting it as a constant makes optimization easier

    // if it's an OR the first expr is emitted in the true branch
    // and the second expr in the false branch, if it's an AND the opposite
    auto get_const_expr = [&] { return graph->insertConstant(is_or, loc); };

    c10::optional<CondValue> rhs;
    auto get_continue_expr = [&] {
      rhs = emitCondExpr(second_expr);
      return rhs->value();
    };

    // if this is an OR, eval second expression if first expr is False
    // If this is an AND, eval second expression if first expr is True
    Value* new_result;
    c10::optional<RefinementSet> refinements;
    c10::optional<bool> static_if;
    if (is_or) {
      new_result = emitIfExpr(loc, lhs, get_const_expr, get_continue_expr);
      refinements = lhs.refinements().Or(rhs->refinements());
      if (lhs.staticIf() && rhs->staticIf()) {
        static_if = *lhs.staticIf() || *rhs->staticIf();
      }
    } else {
      new_result = emitIfExpr(loc, lhs, get_continue_expr, get_const_expr);
      refinements = lhs.refinements().And(rhs->refinements());
      if (lhs.staticIf() && rhs->staticIf()) {
        static_if = *lhs.staticIf() && *rhs->staticIf();
      }
    }
    return CondValue(new_result, std::move(*refinements), static_if);
  }

  Value* emitIfExpr(
      const SourceRange& range,
      const CondValue& cond_value,
      std::function<Value*()> true_expr,
      std::function<Value*()> false_expr) {
    Node* n = graph->insertNode(create(prim::If, range, 0));
    n->addInput(cond_value.value());
    auto* true_block = n->addBlock();
    auto* false_block = n->addBlock();

    auto emit_if_expr = [this, &range](
                            Block* b,
                            const RefinementSet& refinements,
                            std::function<Value*()> expr_value) {
      pushFrame(b);
      WithInsertPoint guard(b);
      insertRefinements(range, refinements);
      Value* out_val = expr_value();
      b->registerOutput(out_val);
      popFrame();
    };

    emit_if_expr(true_block, cond_value.refinements(), std::move(true_expr));
    emit_if_expr(
        false_block, cond_value.refinements().Not(), std::move(false_expr));

    auto true_type = true_block->outputs().at(0)->type();
    auto false_type = false_block->outputs().at(0)->type();
    auto unified = unifyTypes(true_type, false_type);
    if (!unified) {
      throw ErrorReport(range)
          << "if-expression's true branch has type " << true_type->python_str()
          << " but false branch has type " << false_type->python_str();
    }

    // Add op outputs
    auto expr_value = n->addOutput()->setType(*unified); // Resulting value

    return expr_value;
  }
  Value* emitToBool(Value* v) {
    SourceRange loc = v->node()->sourceRange();
    Value* out;
    try {
      auto bool_cast = environment_stack->getSugaredVar("bool", loc);
      out = asSimple(bool_cast->call(loc, method, {v}, {}, 0));
    } catch (...) {
      throw ErrorReport(loc) << "Could not cast value of type "
                             << v->type()->python_str() << " to bool";
    }
    // cast value not response for checking output type
    if (!out->type()->isSubtypeOf(BoolType::get())) {
      throw ErrorReport(loc)
          << "expected a bool expression for condition but found "
          << out->type()->python_str();
    }
    return out;
  }

  void emitIfElseBlocks(
      const SourceRange& loc,
      const CondValue& cond_value,
      const List<Stmt>& trueBranch,
      const List<Stmt>& falseBranch) {
    // this is a static if statement: that is, it contains a subset
    // of operators where we are willing to specialize the if statement
    // to be only the true or false branch when the condition is statically
    // known. This is used to meta-program modules, for instance, when a
    // submodule is absent, an is None check can be used to ensure the
    // accesses to the None check, which would error, are not compiled.
    if (cond_value.staticIf()) {
      if (*cond_value.staticIf()) {
        insertRefinements(loc, cond_value.refinements());
        emitStatements(trueBranch);
      } else {
        insertRefinements(loc, cond_value.refinements().Not());
        emitStatements(falseBranch);
      }
      return;
    }

    Node* n = graph->insertNode(create(prim::If, loc, 0));
    n->addInput(cond_value.value());
    auto* true_block = n->addBlock();
    auto* false_block = n->addBlock();

    // Emit both blocks once to get the union of all mutated values
    auto save_true =
        emitSingleIfBranch(true_block, trueBranch, cond_value.refinements());
    auto save_false = emitSingleIfBranch(
        false_block, falseBranch, cond_value.refinements().Not());

    bool true_exits = exit_blocks.count(true_block);
    bool false_exits = exit_blocks.count(false_block);
    if (true_exits && false_exits) {
      exit_blocks.insert(n->owningBlock());
    }

    // In python, every variable assigned in an if statement escapes
    // the scope of the if statement (all variables are scoped to the function).
    // Script is a subset of python: we consider variables to be in scope
    // as long as there is a definition of the variable along all paths
    // through the if statemnent
    // ----
    // if ...:
    //   a =
    // else:
    //   ...
    // ... = a  # error, a is not defined along all paths
    // ----
    // if ...:
    //   a =
    // else:
    //   a =
    // ... = a # OK, a is defined along all paths
    // ----
    // a = ...
    // if ...:
    //   a =
    // ... = a # OK, a is defined along all paths
    // if ...:
    //   a =
    // else:
    //   return
    // ... = a # OK, a is always defined

    // ordered set, because we want deterministic graph output
    std::set<std::string> mutated_variables;

    // When we access either the true or false environment,
    // we need to set the insertion point so the prim::Load is inserted
    // into the right block.
    // if var is only defined in one branch save error in case it's used later
    for (auto& v : save_true->definedVariables()) {
      {
        WithInsertPoint insert(false_block);
        if (save_false->findInAnyFrame(v) || false_exits) {
          mutated_variables.insert(v);
        } else {
          ErrorReport error(loc);
          environment_stack->setVariableTypeError(v, [=]() -> std::string {
            error << v << " is not defined in the false branch";
            return error.what();
          });
        }
      }
    }
    for (auto& v : save_false->definedVariables()) {
      {
        WithInsertPoint insert(true_block);
        if (save_true->findInAnyFrame(v) || true_exits) {
          mutated_variables.insert(v);
        } else {
          ErrorReport error(loc);
          environment_stack->setVariableTypeError(v, [=]() -> std::string {
            error << v << " is not defined in the true branch";
            return error.what();
          });
        }
      }
    }

    // Register outputs in each block
    for (const auto& x : mutated_variables) {
      Value* tv;
      Value* fv;

      {
        WithInsertPoint insert(true_block);
        if (!true_exits) {
          tv = save_true->getVar(x, loc);
        }
      }
      {
        WithInsertPoint insert(false_block);
        if (!false_exits) {
          fv = save_false->getVar(x, loc);
        }
      }

      // if both branches exit don't emit any variables
      // if one branch exits then we allow the all variables in the other branch
      // to escape scope since they are well-defined
      if (true_exits && false_exits) {
        continue;
      } else if (true_exits) {
        tv = graph->createUninitialized(fv->type())
                 ->insertBefore(true_block->return_node())
                 ->output();
        graph->createStore(x, tv)->insertBefore(true_block->return_node());
      } else if (false_exits) {
        fv = graph->createUninitialized(tv->type())
                 ->insertBefore(false_block->return_node())
                 ->output();
        graph->createStore(x, fv)->insertBefore(false_block->return_node());
      }

      auto unified = unifyTypes(tv->type(), fv->type());

      // attempt to unify the types. we allow variables to be set to different
      // types in each branch as long as that variable is not already in scope,
      // or if that variable does not get used later. here, we save the error
      // so that the error message will be more informative in the case that is
      // used later. When a is accessed in (a + 1), the error will get printed
      // if cond:
      //    a = 1
      // else:
      //    a = tensor
      // b = a + 1
      //
      if (!unified) {
        ErrorReport error(loc);
        error << "Type mismatch: " << x << " is set to type "
              << tv->type()->python_str() << " in the true branch"
              << " and type " << fv->type()->python_str()
              << " in the false branch";
        if (save_true->findInParentFrame(x) ||
            save_false->findInParentFrame(x)) {
          throw error;
        } else {
          environment_stack->setVariableTypeError(
              x, [=]() -> std::string { return error.what(); });
          continue;
        }
      }
      environment_stack->setType(x, *unified);
    }
  }

  CondValue emitIsInstance(Expr obj, Expr classinfo) {
    // turn (float, (int, tuple)) into a flat list of types and type kind
    // category checks: tuple_check = true, types = {float, int}
    struct GatheredTypes {
      GatheredTypes(ScriptTypeParser parser) : typeParser_(std::move(parser)) {}
      void gather(Expr classinfo) {
        if (classinfo.kind() == TK_TUPLE_LITERAL) {
          for (Expr e : TupleLiteral(classinfo).inputs()) {
            gather(e);
          }
          return;
        }
        if (classinfo.kind() == TK_VAR) {
          // Special casing for list and tuple since isinstance(x, list) and
          // isinstance(x, tuple) does not accept List[int] / Tuple[int] like
          // subscript type annotation in python
          auto name = Var(classinfo).name().name();
          if (name == "tuple") {
            tuple_check = true;
            return;
          } else if (name == "list") {
            list_check = true;
            return;
          }
        }
        TypePtr type = typeParser_.parseTypeFromExpr(classinfo);
        types.emplace_back(type);
      }
      bool staticallyTrue(const TypePtr& actual_type) {
        // is this isinstance check statically true?
        if ((list_check && actual_type->kind() == ListType::Kind) ||
            (tuple_check && actual_type->kind() == TupleType::Kind)) {
          return true;
        }
        for (const TypePtr& typ : types) {
          if (actual_type->isSubtypeOf(typ)) {
            return true;
          }
        }
        return false;
      }
      bool maybeOfKind(TypeKind kind, const TypePtr& actual_type) {
        if (actual_type->kind() == AnyType::Kind) {
          return true;
        }
        if (auto op = actual_type->cast<OptionalType>()) {
          return op->getElementType()->kind() == kind;
        }
        return false;
      }
      bool staticallyFalse(const TypePtr& actual_type) {
        if ((list_check && maybeOfKind(ListType::Kind, actual_type)) ||
            (tuple_check && maybeOfKind(TupleType::Kind, actual_type))) {
          return false;
        }
        for (const TypePtr& typ : types) {
          if (typ->isSubtypeOf(actual_type)) {
            return false;
          }
        }
        return true;
      }
      ScriptTypeParser typeParser_;
      bool list_check = false;
      bool tuple_check = false;
      std::vector<TypePtr> types;
    };
    GatheredTypes gathered(typeParser_);
    gathered.gather(classinfo);
    auto val = emitExpr(obj);
    RefinementSet refinement;
    if (gathered.types.size() == 1 && obj.kind() == TK_VAR) {
      std::string ident = Var(obj).name().name();
      Refinement isinstance(
          std::move(ident), gathered.types.at(0));
      refinement = RefinementSet({isinstance}, {});
    }

    if (gathered.staticallyTrue(val->type())) {
      return CondValue(*graph, obj.range(), true, std::move(refinement));
    }
    if (gathered.staticallyFalse(val->type())) {
      return CondValue(*graph, obj.range(), false, std::move(refinement));
    }
    // check maybe true/false at runtime, need an actual op
    Value* result =
        graph
            ->insertNode(graph->createIsInstance(
                val, gathered.types, gathered.list_check, gathered.tuple_check))
            ->output();
    return CondValue(result, std::move(refinement), c10::nullopt);
  }

  void emitIf(const If& stmt) {
    Expr cond = stmt.cond();
    CondValue cond_value = emitCondExpr(cond);
    emitIfElseBlocks(
        stmt.range(), cond_value, stmt.trueBranch(), stmt.falseBranch());
  }

  // *********************** Loop Operators ************************************
  // Emits a loop operator with the form:
  // Loop(max_trip_count)
  // block0(loop_counter) {
  //   <body>
  // }
  // block1 {
  //   <loop condition>
  //   -> (condition)
  // }
  // For loops will have an empty loop condition block with condition set to
  // true. In the convert to ssa pass, the loop condition will correctly
  // inlined. and inputs and outputs added so that the loop conforms to the
  // semantics specified at
  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Loop
  void emitLoopCommon(
      SourceRange range,
      const List<Stmt>& body,
      const SugaredValuePtr& iter_val,
      c10::optional<List<Expr>> targets,
      c10::optional<Expr> cond) {
    Value* max_trip_count_val = nullptr;
    if (iter_val != nullptr) {
      max_trip_count_val = iter_val->len(range, method);
    } else {
      max_trip_count_val = materializeConstant(
          std::numeric_limits<int64_t>::max(),
          *graph,
          range,
          integral_constants);
    }

    Node* n = graph->insertNode(create(prim::Loop, range, 0));
    auto* body_block = n->addBlock();
    {
      Block* condition_block = n->addBlock();
      pushFrame(condition_block);
      Value* out;
      if (cond) {
        WithInsertPoint insert(condition_block);
        out = emitToBool(emitExpr(cond.value()));
      } else {
        WithInsertPoint insert(n);
        out = graph->insertConstant(true, range);
      }
      condition_block->registerOutput(out);
      popFrame();
    }
    n->addInput(max_trip_count_val);

    Value* trip_count =
        body_block->addInput()->setType(IntType::get()); // Iteration num
    {
      pushFrame(body_block);
      WithInsertPoint guard(body_block);

      // if the FOR iters and targets are present, emit FOR target assignments
      if (iter_val != nullptr && targets) {
        Value* cur_elem = iter_val->getitem(range, method, trip_count);
        SugaredValuePtr sv = std::make_shared<SimpleValue>(cur_elem);
        List<Expr> target_exprs = targets.value();
        validateAssignLhsExpr(target_exprs, range);

        // if target exprs are more than 1, it means iteration unpacking on LHS
        // we create Tuple literal to wrap those target exprs for assignments
        if (target_exprs.size() > 1) {
          Expr tl = TupleLiteral::create(range, target_exprs);
          target_exprs = List<Expr>::create(range, {tl});
        }
        emitExprsAssign(target_exprs, {sv}, range, /*n_binders=*/1);
      }

      emitStatements(body);

      popFrame();
    }
  }

  void emitFor(const For& stmt) {
    auto targets = stmt.targets();
    auto itrs = stmt.itrs();
    auto body = stmt.body();
    if (stmt.itrs().size() != 1) {
      throw ErrorReport(stmt) << "List of iterables is not supported currently";
    }
    // Emit loop information for builtinFunction values like range(), zip(),
    // enumerate() or SimpleValue like List, Tensor, Dict, etc.
    SugaredValuePtr sv = emitSugaredExpr(itrs[0], 1);

    // We will get IterableTree for builtinFunctions zip() and enumerate(),
    // RangeValue for range(), and SimpleValue for types like
    // List/Tensor/Dict/String.
    auto range_val = std::dynamic_pointer_cast<RangeValue>(sv);
    auto siv = std::dynamic_pointer_cast<SimpleValue>(sv);
    auto iterable_tree = std::dynamic_pointer_cast<IterableTree>(sv);

    // For SimpleValue(except Tuple) or RanveValue/IterableTree, emit common
    // loop
    if ((siv && !siv->getValue()->type()->cast<TupleType>()) || range_val ||
        iterable_tree) {
      // looping over a dict defaults to looping over the keys in python
      if (siv && siv->getValue()->type()->cast<DictType>()) {
        sv = std::make_shared<SimpleValue>(
            graph->insert(aten::keys, {siv->getValue()}, {}, stmt.range()));
      }
      emitLoopCommon(stmt.range(), body, sv, targets, {});
      return;
    }

    // Emit or unroll the loop for Tuple or ModuleList, we choose to unroll or
    // emit each subelemnt for each iteration separately. This is because for
    // ModuleList, each module inside the list may be different types, so FOR ..
    // in ModuleList essentially should emit different stmts for each iteration,
    // which we shouldn't emit the prim::Loop node for it, the same rule applies
    // for the Tuple case.
    auto instances = sv->asTuple(stmt.range(), method);
    pushFrame(environment_stack->block());
    for (const auto& inst : instances) {
      emitExprsAssign(targets, {inst}, itrs[0].range(), /*n_binders=*/1);
      emitStatements(body);
    }

    for (const auto& n : environment_stack->definedVariables()) {
      if (environment_stack->findInParentFrame(n)) {
        environment_stack->next->setVar(
            stmt.range(), n, environment_stack->getVar(n, stmt.range()));
      }
    }
    popFrame();
  }

  void emitWhile(const While& stmt) {
    auto cond = stmt.cond();
    emitLoopCommon(stmt.range(), stmt.body(), nullptr, {}, cond);
  }

  // Currently we do not support assigning exceptions to variables,
  // a = Exception("hi")
  // raise a
  //
  // We ignore the expression following raise
  void emitRaise(const SourceRange& loc) {
    const std::string exception = "Exception";
    auto string_input = insertConstant(*graph, exception, loc);
    graph->insert(prim::RaiseException, {string_input}, {}, loc);
    exit_blocks.insert(environment_stack->block());
  }

  // emit assserions as an if branch so that assertions will reuse the
  void emitAssert(const Assert& stmt) {
    CondValue cond_value = emitCondExpr(stmt.test());
    List<Stmt> true_branch = List<Stmt>::create(stmt.range(), {});
    List<Stmt> false_branch =
        List<Stmt>::create(stmt.range(), {Raise::create(stmt.range())});
    emitIfElseBlocks(stmt.range(), cond_value, true_branch, false_branch);
  }

  // Validate that the `lhs` Expr's in an assignment statement are valid. That
  // is:
  //
  // 1) All lhs Expr's are either Var, Tuple or Starred nodes
  // 2) There is at most one Starred node in the lhs Expr
  // 3) A Starred node can only appear when there is another non-Starred lhs
  //    Expr. Concretely this means that `*abc = func()` is illegal. Unpacking
  //    all outputs into a tuple is covered by `abc = func()`.
  bool validateAssignLhsExpr(const List<Expr>& lhs, const SourceRange& r) {
    size_t num_normal_assign = 0;
    size_t num_starred = 0;
    for (const auto& assignee : lhs) {
      if (assignee.kind() == TK_VAR || assignee.kind() == TK_SUBSCRIPT ||
          assignee.kind() == TK_TUPLE_LITERAL) {
        num_normal_assign++;
      } else if (assignee.kind() == TK_STARRED) {
        num_starred++;
      } else {
        throw ErrorReport(assignee) << "lhs of assignment must be a variable, "
                                    << "subscript, or starred expression";
      }
    }

    if (num_starred > 1) {
      throw ErrorReport(r)
          << "Only one starred expression is allowed on the lhs";
    }

    if (num_starred > 0 && num_normal_assign == 0) {
      throw ErrorReport(r) << "A Starred expression may only appear on the "
                           << "lhs within the presence of another non-starred"
                           << " expression";
    }

    return num_starred;
  }

  // Get the appropriate builtin op for this augmented assignment
  // If the RHS is a tensor, return the corresponding ATen in-place op
  // If it's a list of scalars, then return the corresponding list augment op
  Symbol getAugOp(const AugAssign& stmt, const TypePtr& type) {
    if (type->cast<ListType>()) { // Lists also have in-place ops.
      switch (stmt.aug_op()) {
        case '+':
          return aten::add_;
      }
    }
    bool isTensor = type->isSubtypeOf(TensorType::get());
    switch (stmt.aug_op()) {
      case '+':
        return isTensor ? aten::add_ : aten::add;
      case '-':
        return isTensor ? aten::sub_ : aten::sub;
      case '/':
        return isTensor ? aten::div_ : aten::div;
      case '*':
        return isTensor ? aten::mul_ : aten::mul;
      default:
        throw ErrorReport(stmt)
            << "Unknown augmented assignment: " << kindToString(stmt.aug_op());
    }
  }

  // Emit nodes for augmented assignments like `+=`
  void emitAugAssignment(const AugAssign& stmt) {
    switch (stmt.lhs().kind()) {
      case TK_VAR: {
        emitAugAssignmentToVar(stmt);
      } break;
      case '.': {
        emitAugAssignmentToSelectVar(stmt);
      } break;
      case TK_SUBSCRIPT: {
        emitAugAssignmentToSubscript(stmt);
      } break;
      default:
        throw ErrorReport(stmt.lhs())
            << "unexpected expression on "
            << "left-hand side of augmented assignment";
    }
  }

  // This will be called when there is a class param or module buffer
  // mutation which make the LHS of the expr be a select expression
  //
  // Example like:
  // class A(Module):
  //  def __init__():
  //    self.register_buffer("running_var", torch.zeros(1))
  //
  //  def forward():
  //    self.num_batches += 1
  //
  // In this case we will only consider the scenario that the module
  // buffer type is a tensor, and we emit the corresponding tensor
  // in place op, and throw error for other unsupported types
  void emitAugAssignmentToSelectVar(const AugAssign& stmt) {
    const auto lhs = Select(stmt.lhs());
    const auto lhsSugaredVar =
        environment_stack->getSugaredVar(Var(lhs.value()).name());
    const auto lhsValue =
        lhsSugaredVar->attr(lhs.range(), method, lhs.selector().name())
            ->asValue(lhs.range(), method);
    if (lhsValue->type()->isSubtypeOf(TensorType::get())) {
      // for module parameter/buffer assignment, only consider tensor types,
      // emit the corresponding in-place op
      const auto rhs = NamedValue(stmt.rhs().range(), emitExpr(stmt.rhs()));
      const auto self = NamedValue(stmt.lhs().range(), "self", lhsValue);
      emitBuiltinCall(
          stmt.range(),
          *method.graph(),
          getAugOp(stmt, lhsValue->type()),
          self,
          {rhs},
          {},
          /*required=*/true);

    } else {
      throw ErrorReport(stmt.lhs())
          << "left-hand side of augmented assignment to module "
          << "parameters/buffers can only be tensor types";
    }
  }

  void emitAugAssignmentToVar(const AugAssign& stmt) {
    const auto lhs = Var(stmt.lhs());
    const auto lhsValue = environment_stack->getSugaredVar(lhs.name())
                              ->asValue(lhs.range(), method);
    auto lhsType = lhsValue->type();
    if (lhsType->isSubtypeOf(TensorType::get()) ||
        lhsType->cast<c10::ListType>()) {
      // for tensors, emit the corresponding in-place op
      const auto rhs = NamedValue(stmt.rhs().range(), emitExpr(stmt.rhs()));
      const auto self = NamedValue(stmt.lhs().range(), "self", lhsValue);
      const auto output = emitBuiltinCall(
          stmt.range(),
          *method.graph(),
          getAugOp(stmt, lhsValue->type()),
          self,
          {rhs},
          {},
          /*required=*/true);

      environment_stack->setVar(lhs.range(), lhs.name().name(), output);
    } else {
      // for primitive types, desugar into a simple assignment
      //   e.g. foo += 1 becomes foo.2 = foo + 1
      Ident lhs = Var(stmt.lhs()).name();
      Expr expr = BinOp::create(
          stmt.range(),
          stmt.aug_op(),
          Var::create(lhs.range(), lhs),
          stmt.rhs());
      environment_stack->setVar(lhs.range(), lhs.name(), emitExpr(expr));
    }
  }

  void emitAugAssignmentGeneric(
      const AugAssign& stmt,
      const Subscript& lhs,
      Value* sliceable) {
    // Get the idx to augment
    const auto subscriptExprs = lhs.subscript_exprs();
    const TypePtr type = sliceable->type();
    if (subscriptExprs.size() != 1) {
      throw ErrorReport(subscriptExprs)
          << "Sliced expression not yet supported for " << type->python_str()
          << " augmented assignment. "
          << "File a bug if you want this";
    }

    TypePtr elemType = nullptr;
    if (const ListTypePtr listType = type->cast<ListType>()) {
      elemType = listType->getElementType();
    } else if (const DictTypePtr dictType = type->cast<DictType>()) {
      elemType = dictType->getKeyType();
    }

    if (elemType == nullptr) {
      throw ErrorReport(lhs)
          << type->python_str() << " does not support augmented assignment.";
    }
    const auto idxValue = emitExpr(subscriptExprs[0]);
    const auto containerArg =
        NamedValue(lhs.value().range(), type->str(), sliceable);
    const auto idxArg = NamedValue(subscriptExprs.range(), "idx", idxValue);
    const auto valueArg =
        NamedValue(stmt.rhs().range(), "value", emitExpr(stmt.rhs()));

    const auto getItem = graph->insert(
        aten::__getitem__, {containerArg, idxArg}, {}, stmt.range());
    const auto augmentedItem = graph->insert(
        getAugOp(stmt, elemType), {getItem, valueArg}, {}, stmt.range());
    graph->insert(
        aten::_set_item,
        {containerArg, idxArg, augmentedItem},
        {},
        stmt.range());
  }

  void emitAugAssignmentToSubscript(const AugAssign& stmt) {
    // Process the base list value
    const auto lhs = Subscript(stmt.lhs());
    const auto sliceable = emitExpr(lhs.value());

    if (sliceable->type()->isSubtypeOf(TensorType::get())) {
      // If it's a tensor, just fully evaluate the subscript operation and emit
      // an in-place assignment
      std::vector<Value*> tensorIndices;
      Value* sliced;
      std::tie(sliced, tensorIndices) = emitIntAndSliceIndexing(
          lhs.range(), sliceable, lhs.subscript_exprs());

      const auto slicedArg = NamedValue(stmt.lhs().range(), "self", sliced);
      const auto rhs = NamedValue(stmt.rhs().range(), emitExpr(stmt.rhs()));
      if (tensorIndices.size() == 0) {
        // Common case: we only tried to index with int and slices. Emit the
        // correct augmented assignment op to the sliced value
        emitBuiltinCall(
            stmt.range(),
            *method.graph(),
            getAugOp(stmt, sliceable->type()),
            slicedArg,
            {rhs},
            {},
            /*required=*/true);
      } else {
        // Special case: we tried to do "advanced indexing". Lower this expr
        // into `index` and `index_put_` ops with tensordices of Tensor?[]
        const auto indices = graph
                                 ->insertNode(graph->createList(
                                     OptionalType::ofTensor(), tensorIndices))
                                 ->output();
        const auto indexed =
            graph->insert(aten::index, {slicedArg, indices}, {}, stmt.range());
        const auto augmented = emitBuiltinCall(
            stmt.range(),
            *method.graph(),
            getAugOp(stmt, sliceable->type()),
            indexed,
            {rhs},
            {},
            /*required=*/true);
        graph->insert(
            aten::index_put_,
            {slicedArg, indices, augmented},
            {},
            stmt.range());
      }
    } else {
      emitAugAssignmentGeneric(stmt, lhs, sliceable);
    }
  }

  // Emit mutating assignments like `foo[0] = bar`
  void emitSubscriptAssign(
      const SourceRange& stmtRange,
      const Subscript& lhs,
      const Expr& rhs) {
    emitSubscriptAssign(stmtRange, lhs, NamedValue(rhs.range(), emitExpr(rhs)));
  }

  void emitSubscriptAssign(
      const SourceRange& stmtRange,
      const Subscript& lhs,
      const NamedValue& rhs) {
    // First check the base value.
    auto sliceable = emitExpr(lhs.value());

    // If it's a tensor, copy the RHS data into it
    if (sliceable->type()->isSubtypeOf(TensorType::get())) {
      std::vector<Value*> tensorIndices;
      Value* sliced;
      // Handle multi-dimensional slicing: first emit int/slice indexing
      // TODO: the Python equivalent code has special-cased copy_to
      // broadcasting to match NumPy semantics (see PR#4853). We can't
      // replicate that without knowing the size of the Tensor; so really that
      // code should be moved into the aten function
      std::tie(sliced, tensorIndices) = emitIntAndSliceIndexing(
          lhs.range(), sliceable, lhs.subscript_exprs());

      const auto slicedArg = NamedValue(lhs.range(), sliced);
      if (tensorIndices.size() == 0) {
        // Common case: we only tried to index with int and slices. Copy the
        // RHS into the resulting tensor.
        graph->insert(aten::copy_, {slicedArg, rhs}, {}, stmtRange);
      } else {
        // Special case: we tried to do "advanced indexing" with a tensor.
        // Dispatch to `aten::index_put_` with tensorindices of Tensor?[]
        const auto indices = graph
                                 ->insertNode(graph->createList(
                                     OptionalType::ofTensor(), tensorIndices))
                                 ->output();

        graph->insert(
            aten::index_put_, {slicedArg, indices, rhs}, {}, stmtRange);
      }
      // Otherwise, this is a list or a classtype.
      // Dispatch to aten::_set_item to both select and assign
    } else {
      const auto subscript = lhs.subscript_exprs();
      if (subscript.size() != 1 || subscript[0].kind() == TK_SLICE_EXPR) {
        throw ErrorReport(subscript)
            << "Sliced expression not yet supported for"
            << " subscripted assignment. "
            << "File a bug if you want this";
      }

      std::vector<NamedValue> args;
      args.emplace_back(lhs.value().range(), "self", sliceable);
      args.emplace_back(
          lhs.subscript_exprs().range(), "idx", emitExpr(subscript[0]));
      args.push_back(rhs);
      makeMagic(
          "__setitem__",
          std::make_shared<BuiltinFunction>(aten::_set_item, at::nullopt))
          ->call(stmtRange, method, args, {}, 0);
    }
  }

  void emitTupleAssign(const TupleLiteral& tl, const Expr& rhs) {
    size_t n_binders = tl.inputs().size();
    bool starred_unpack = validateAssignLhsExpr(tl.inputs(), tl.range());
    if (starred_unpack)
      n_binders--;
    auto output = emitSugaredExpr(rhs, n_binders);
    emitTupleAssign(tl, output, rhs.range(), n_binders, starred_unpack);
  }

  void emitTupleAssign(
      const TupleLiteral& tl,
      const SugaredValuePtr& rhs_output,
      const SourceRange& rhs_loc,
      size_t n_binders,
      bool starred_unpack) {
    auto outputs = rhs_output->asTuple(
        rhs_loc,
        method,
        starred_unpack ? c10::nullopt : c10::optional<size_t>{n_binders});
    if (outputs.size() < n_binders) {
      throw ErrorReport(tl)
          << "need " << (starred_unpack ? "at least " : "") << n_binders
          << " values to unpack but found only " << outputs.size();
    }
    if (outputs.size() > n_binders && !starred_unpack) {
      throw ErrorReport(tl) << "too many values to unpack: need " << n_binders
                            << " but found " << outputs.size();
    }

    emitExprsAssign(tl.inputs(), outputs, rhs_loc, n_binders);
  }

  void emitExprsAssign(
      const List<Expr>& lhs_exprs,
      const at::ArrayRef<SugaredValuePtr> outputs,
      const SourceRange& rhs_loc,
      size_t n_binders) {
    int i = 0;
    for (auto assignee : lhs_exprs) {
      switch (assignee.kind()) {
        case TK_SUBSCRIPT:
          emitSubscriptAssign(
              rhs_loc,
              Subscript(assignee),
              NamedValue(rhs_loc, outputs.at(i)->asValue(rhs_loc, method)));
          i++;
          break;
        case TK_VAR:
          environment_stack->setSugaredVar(
              assignee.range(),
              Var(assignee).name().name(),
              outputs.at(i),
              /*annotated_type=*/nullptr);
          i++;
          break;
        case TK_STARRED: {
          auto var = Starred(assignee).expr();
          if (var.kind() != TK_VAR) {
            throw ErrorReport(var) << "Cannot pack a tuple into a non-variable";
          }
          size_t n_matched = outputs.size() - n_binders;
          ArrayRef<std::shared_ptr<SugaredValue>> outputs_ref = outputs;
          auto values = fmap(
              outputs_ref.slice(i, n_matched),
              [&](const std::shared_ptr<SugaredValue>& v) {
                return v->asValue(assignee.range(), method);
              });
          auto tup = graph->insertNode(graph->createTuple(values))->output();
          environment_stack->setVar(var.range(), Var(var).name().name(), tup);
          i += n_matched;
        } break;
        case TK_TUPLE_LITERAL: {
          // recursively emit tuple assignments on tuple literal input
          TupleLiteral sub_tl = TupleLiteral(assignee);
          size_t sub_n_binders = sub_tl.inputs().size();
          bool sub_starred_unpack =
              validateAssignLhsExpr(sub_tl.inputs(), sub_tl.range());
          if (sub_starred_unpack)
            sub_n_binders--;
          emitTupleAssign(
              sub_tl,
              outputs.at(i),
              rhs_loc,
              sub_n_binders,
              sub_starred_unpack);
          i++;
        } break;
        default:
          throw ErrorReport(assignee)
              << "unexpected expression on the left-hand side";
      }
    }
  }

  void emitAssignment(const Assign& stmt) {
    if (stmt.lhs_list().size() == 1) {
      return emitSingleAssignment(stmt);
    }
    // multiple assign & annotated type not supported in python
    TORCH_INTERNAL_ASSERT(stmt.lhs_list().size() > 1 && !stmt.type().present());
    // a = b = expr()
    // the semantics of multiple assignment is that expr() is emitted once, then
    // from left to right the assignments are made
    const auto tmp_name = createTempName("$tmp_assign_");
    environment_stack->setSugaredVar(
        stmt.rhs().range(),
        tmp_name,
        emitSugaredExpr(stmt.rhs().get(), 1),
        /*annotated_type=*/nullptr);
    auto ident = Var::create(
        stmt.rhs().range(), Ident::create(stmt.rhs().range(), tmp_name));
    for (auto expr : stmt.lhs_list()) {
      emitSingleAssignment(Assign::create(
          stmt.range(),
          List<Expr>::create(expr.range(), {expr}),
          Maybe<Expr>::create(stmt.rhs().range(), ident),
          Maybe<Expr>::create(stmt.range())));
    }
  }

  void emitSingleAssignment(const Assign& stmt) {
    if (!stmt.rhs().present()) {
      throw ErrorReport(stmt.range())
          << "For an assignment, expected an expression on the right-hand side";
    }
    const Expr& rhs = stmt.rhs().get();
    switch (stmt.lhs().kind()) {
      case TK_VAR: {
        auto v = Var(stmt.lhs());
        TypePtr type = nullptr;
        if (stmt.type().present()) {
          type = typeParser_.parseTypeFromExpr(stmt.type().get());
        }
        environment_stack->setSugaredVar(
            v.range(),
            v.name().name(),
            emitSugaredExpr(rhs, 1, type),
            /*annotated_type=*/type);
      } break;
      case TK_TUPLE_LITERAL:
        emitTupleAssign(TupleLiteral(stmt.lhs()), rhs);
        break;
      case '.':
        emitSelectAssign(stmt);
        break;
      case TK_SUBSCRIPT:
        emitSubscriptAssign(stmt.range(), Subscript(stmt.lhs()), rhs);
        break;
      default:
        throw ErrorReport(stmt.lhs())
            << "unexpected expression on left-hand side of assignment";
    }
  }

  void emitSelectAssign(const Assign& stmt) {
    if (!stmt.rhs().present()) {
      throw ErrorReport(stmt.range()) << "Expected RHS for assignment";
    }
    const auto lhs = Select(stmt.lhs());
    const auto basename = Var(lhs.value()).name();
    const auto rhsValue = emitSugaredExpr(stmt.rhs().get(), 1)
                              ->asValue(stmt.rhs().range(), method);
    auto userObject = environment_stack->getSugaredVar(basename);
    userObject->setAttr(stmt.range(), method, lhs.selector().name(), rhsValue);
  }

  NodeKind getNodeKind(int kind, int ninputs) {
    switch (kind) {
      case '+':
        return aten::add;
      case '-':
        return aten::sub;
      case TK_UNARY_MINUS:
        return aten::neg;
      case '*':
        return aten::mul;
      case TK_POW:
        return aten::pow;
      case '@':
        return aten::matmul;
      case TK_STARRED:
        return prim::Starred;
      case '/':
        return aten::div;
      case '%':
        return aten::remainder;
      case TK_NE:
        return aten::ne;
      case TK_EQ:
        return aten::eq;
      case '<':
        return aten::lt;
      case '>':
        return aten::gt;
      case TK_LE:
        return aten::le;
      case TK_GE:
        return aten::ge;
      case TK_AND:
        return aten::__and__;
      case TK_OR:
        return aten::__or__;
      case TK_IS:
        return aten::__is__;
      case TK_ISNOT:
        return aten::__isnot__;
      case TK_NOT:
        return aten::__not__;
      case TK_FLOOR_DIV:
        return aten::floordiv;
      case '&':
        return aten::__and__;
      case '|':
        return aten::__or__;
      case '^':
        return aten::__xor__;
      case TK_IN:
        return aten::__contains__;
      default:
        throw std::runtime_error("unknown kind " + std::to_string(kind));
    }
  }

  std::string getOperatorOverload(int kind, int ninputs) {
    switch (kind) {
      case '+':
        return "__add__";
      case '-':
        return "__sub__";
      case TK_UNARY_MINUS:
        return "__neg__";
      case '~':
        return "__invert__";
      case '*':
        return "__mul__";
      case TK_POW:
        return "__pow__";
      case '/':
        return "__truediv__";
      case '%':
        return "__mod__";
      case TK_NE:
        return "__ne__";
      case TK_EQ:
        return "__eq__";
      case '<':
        return "__lt__";
      case '>':
        return "__gt__";
      case TK_LE:
        return "__le__";
      case TK_GE:
        return "__ge__";
      case '&':
        return "__and__";
      case '|':
        return "__or__";
      case '^':
        return "__xor__";
      case TK_IN:
        return "__contains__";
      default:
        throw std::runtime_error("unknown kind " + std::to_string(kind));
    }
  }

  std::vector<NamedValue> getNamedValues(
      const TreeList& trees,
      bool maybe_unpack) {
    std::vector<NamedValue> values;
    for (const auto& tree : trees) {
      if (maybe_unpack && tree->kind() == TK_STARRED) {
        auto starred = Starred(tree);
        auto entries = emitSugaredExpr(starred.expr(), 1)
                           ->asTuple(starred.range(), method);
        for (const auto& entry : entries) {
          values.emplace_back(
              tree->range(), entry->asValue(starred.range(), method));
        }
      } else {
        values.emplace_back(tree->range(), emitExpr(Expr(tree)));
      }
    }
    return values;
  }
  std::vector<NamedValue> getNamedValues(
      const List<Expr>& trees,
      bool maybe_unpack) {
    return getNamedValues(trees.tree()->trees(), maybe_unpack);
  }

  std::vector<Value*> getValues(const TreeList& trees, bool maybe_unpack) {
    return toValues(*graph, getNamedValues(trees, maybe_unpack));
  }
  std::vector<Value*> getValues(const List<Expr>& trees, bool maybe_unpack) {
    return getValues(trees.tree()->trees(), maybe_unpack);
  }

  std::vector<NamedValue> emitAttributes(const List<Attribute>& attributes) {
    return fmap(attributes, [&](const Attribute& attr) {
      return NamedValue(
          attr.range(), attr.name().name(), emitExpr(attr.value()));
    });
  }

  void checkApplyNumInputs(Apply& apply, size_t expected_inputs) {
    const SourceRange& loc = apply.range();
    if (apply.inputs().size() != expected_inputs) {
      throw ErrorReport(loc)
          << Var(apply.callee()).name().name() << " expected exactly "
          << expected_inputs << " arguments but found "
          << apply.inputs().size();
    }
    if (apply.attributes().size() > 0) {
      throw ErrorReport(loc)
          << Var(apply.callee()).name().name() << " takes no keyword arguments";
    }
  }

  std::shared_ptr<SugaredValue> emitApplyExpr(Apply& apply, size_t n_binders) {
    auto sv = emitSugaredExpr(apply.callee(), 1);
    auto loc = apply.callee().range();
    if (auto fork_value = dynamic_cast<ForkValue*>(sv.get())) {
      auto& trees = apply.inputs().tree()->trees();
      if (trees.size() < 1) {
        throw ErrorReport(loc) << "Expected at least one argument to fork()";
      }
      auto forked = emitSugaredExpr(Expr(trees[0]), 1);
      TreeList sliced_trees(trees.begin() + 1, trees.end());
      auto inputs = getNamedValues(sliced_trees, true);
      auto attributes = emitAttributes(apply.attributes());
      return emitForkExpr(loc, forked, inputs, attributes);
    } else if (auto annotate_value = dynamic_cast<AnnotateValue*>(sv.get())) {
      checkApplyNumInputs(apply, 2);
      TypePtr type = typeParser_.parseTypeFromExpr(apply.inputs()[0]);
      Value* expr = tryConvertToType(
          apply.range(),
          *graph,
          type,
          emitExpr(apply.inputs()[1], type),
          /*allow_conversions=*/true);

      std::stringstream why_not;
      if (!expr->type()->isSubtypeOfExt(type, &why_not)) {
        throw ErrorReport(apply.inputs())
            << "expected an expression of type " << type->python_str()
            << " but found " << expr->type()->python_str() << "\n"
            << why_not.str();
      }

      // None is a subtype of Optional[T], but we want to remember what T is,
      // after annotation so that variables assigned to this None will still
      // get the right type. To do this, we make a None constant that
      // has the type Optional[T]
      if (type->kind() == OptionalType::Kind &&
          expr->type()->isSubtypeOf(NoneType::get())) {
        Node* none = graph->createNone();
        none->output()->setType(type);
        graph->insertNode(none);
        expr = none->output();
      }

      return std::make_shared<SimpleValue>(expr);
    } else if (auto getattr = dynamic_cast<GetAttrValue*>(sv.get())) {
      checkApplyNumInputs(apply, 2);
      auto obj = emitSugaredExpr(apply.inputs()[0], 1);
      auto selector = apply.inputs()[1];
      if (selector.kind() != TK_STRINGLITERAL) {
        throw ErrorReport(loc)
            << "getattr's second argument must be a string literal";
      }
      const std::string& name = StringLiteral(selector).text();
      return obj->attr(apply.range(), method, name);
    } else if (
        auto uninitialized_value =
            dynamic_cast<UninitializedValue*>(sv.get())) {
      checkApplyNumInputs(apply, 1);
      TypePtr type = typeParser_.parseTypeFromExpr(apply.inputs()[0]);
      auto out = graph->insertNode(graph->createUninitialized(type))
                     ->setSourceRange(loc);
      return std::make_shared<SimpleValue>(out->output());
    } else if (auto tuple_call = dynamic_cast<TupleCallValue*>(sv.get())) {
      checkApplyNumInputs(apply, 1);
      auto arg = emitSugaredExpr(apply.inputs()[0], 1);
      auto inputs = arg->asTuple(apply.range(), method);
      auto inp_values = fmap(inputs, [&](const SugaredValuePtr& sv) {
        return sv->asValue(loc, method);
      });
      return std::make_shared<SimpleValue>(
          graph->insertNode(graph->createTuple(inp_values))->output());
    } else if (auto isinstance = dynamic_cast<IsInstanceValue*>(sv.get())) {
      checkApplyNumInputs(apply, 2);
      auto result = emitIsInstance(apply.inputs()[0], apply.inputs()[1]);
      return std::make_shared<SimpleValue>(result.value());
    } else if (auto classNew = dynamic_cast<ClassNewMethod*>(sv.get())) {
      if (apply.inputs().size() != 1) {
        throw ErrorReport(loc) << "Only one argument to __new__ allowed";
      }
      auto arg = emitSugaredExpr(apply.inputs()[0], 1);
      auto class_arg = dynamic_cast<ClassValue*>(arg.get());
      if (!class_arg) {
        throw ErrorReport(loc)
            << "Expected class value as argument to __new__, got "
            << arg->kind() << " instead";
      }
      if (class_arg->type_ != classNew->type_) {
        throw ErrorReport(loc)
            << "Argument to __new__() must match the class "
            << "you are calling __new__() on. "
            << "Got: " << class_arg->type_->python_str()
            << ", expected: " << classNew->type_->python_str();
      }

      return classNew->createObject(apply.range(), method);
    } else if (auto iterable = std::dynamic_pointer_cast<IterableValue>(sv)) {
      return emitIterableTree(loc, apply.inputs(), iterable);
    } else {
      auto inputs = getNamedValues(apply.inputs(), true);
      auto attributes = emitAttributes(apply.attributes());
      return sv->call(loc, method, inputs, attributes, n_binders);
    }
  }

  Value* emitExpr(const Expr& tree, const TypePtr& type_hint = nullptr) {
    // Push the source range of a call in case compiling this function
    // triggers an error
    ErrorReport::CallStack::update_pending_range(tree.range());
    return emitSugaredExpr(tree, 1, type_hint)->asValue(tree.range(), method);
  }

  NodeKind reverseComparision(NodeKind kind) {
    if (kind == aten::lt) {
      return aten::gt;
    } else if (kind == aten::le) {
      return aten::ge;
    } else if (kind == aten::gt) {
      return aten::lt;
    } else if (kind == aten::ge) {
      return aten::le;
    }
    throw std::runtime_error(
        "reverseComparision: unsupported NodeKind. File a bug");
  }

  // any expression that can produce a SugaredValue is handled here
  // expressions that only return a single Value* are handled in emitSimpleExpr
  // type_hint is set if there is a type that this value is expected to be
  // e.g. a : List[int] = []
  // or a = torch.jit.annotate(List[int], [])
  // the caller is responsible for checking that the result matches type_hint
  // emitSugaredExpr is free to ignore it.
  std::shared_ptr<SugaredValue> emitSugaredExpr(
      const Expr& tree,
      size_t n_binders,
      const TypePtr& type_hint = nullptr) {
    switch (tree.kind()) {
      case TK_VAR:
        return environment_stack->getSugaredVar(Var(tree).name());
      case '.': {
        auto select = Select(tree);
        auto sv = emitSugaredExpr(select.value(), 1);
        return sv->attr(select.range(), method, select.selector().name());
      }
      case TK_APPLY: {
        auto apply = Apply(tree);
        return emitApplyExpr(apply, n_binders);
      } break;
      default:
        return std::make_shared<SimpleValue>(emitSimpleExpr(tree, type_hint));
    }
  }

  Value* emitUnaryOp(const TreeRef& tree, const std::string &magicMethod, const c10::Symbol &opSymbol) {
    const auto& inputs = tree->trees();
    auto named_values = getNamedValues(inputs, /*maybe_unpack=*/false);
    auto val =
        asSimple(makeMagic(
                     magicMethod,
                     std::make_shared<BuiltinFunction>(opSymbol, at::nullopt))
                     ->call(tree->range(), method, named_values, {}, 0));

    // if we emitted the unary op and not some other overloaded function,
    // then try to constantfold
    if (val->node()->kind() != opSymbol) {
      return val;
    }
    auto maybe_constant_input = toIValue(val->node()->input());
    if (!maybe_constant_input) {
      return val;
    }
    auto op = getOperation(val->node());
    Stack stack;
    stack.push_back(*maybe_constant_input);
    op(stack);
    AT_ASSERT(stack.size() == 1);
    return graph->insertConstant(stack[0], tree->range());
  }


  // We construct the iterable tree here using the IterableTree SugaredValue,
  // The tree consists of SimpleValue, RangeValue or IterableValue:
  // For SimpleValues(List, Dict, etc) or RangeValue. We will make them as tree
  // leaves since we could get the loop information from len() and get_item().
  // For IterableValue like zip(), enumerate(), we can model them as a
  // combination of leaves, and we emit a IterableTree value to record the tree
  // information
  SugaredValuePtr emitIterableTree(
      SourceRange& loc,
      const List<Expr>& inputs,
      const std::shared_ptr<IterableValue>& iterable) {
    std::shared_ptr<IterableTree> iterable_tree = nullptr;
    size_t input_size = inputs.size();

    // Handling different iterable values
    if (iterable->symbol_ == prim::range) {
      std::vector<Value*> input_vals = getValues(inputs, /*maybe_unpack=*/true);
      return std::make_shared<RangeValue>(loc, method, input_vals);
    } else if (iterable->symbol_ == prim::enumerate) {
      // enumerate(x) can be rewrite as subtrees:
      // IterableTree(RangeValue(0, math.inf), SimpleValue(x))
      Value* start_index = nullptr;
      if (input_size == 0) {
        throw ErrorReport(loc)
            << "enumerate expected at least 1 arguments, got 0";
      }

      if (input_size == 2) {
        start_index = emitSugaredExpr(inputs[1], 1)->asValue(loc, method);
      }

      if (input_size > 2) {
        throw ErrorReport(loc)
            << "enumerate expected at most 2 arguments, got " << input_size;
      }
      std::vector<Value*> range_inputs;
      if (start_index != nullptr) {
        range_inputs.emplace_back(start_index);
      }
      Value* end = materializeConstant(
          std::numeric_limits<int64_t>::max(), *graph, loc, integral_constants);
      range_inputs.emplace_back(end);
      SugaredValuePtr range_sv =
          std::make_shared<RangeValue>(loc, method, range_inputs);
      SugaredValuePtr expr_sv = emitSugaredExpr(inputs[0], 1);
      iterable_tree = std::make_shared<IterableTree>(
          std::vector<SugaredValuePtr>({range_sv, expr_sv}));
    } else if (iterable->symbol_ == prim::zip) {
      // zip(x, y) can be rewrite as subtrees:
      // IterableTree(IterableTree(x), IterableTree(y))
      if (inputs.size() == 0) {
        throw ErrorReport(loc) << "zip expected at least 1 arguments, got 0";
      }
      iterable_tree = std::make_shared<IterableTree>();
      for (Expr expr : inputs) {
        auto expr_sv = emitSugaredExpr(expr, 1);
        iterable_tree->addChild(expr_sv);
      }
    }
    return iterable_tree;
  }

  std::shared_ptr<SugaredValue> emitForkExpr(
      SourceRange loc,
      const std::shared_ptr<SugaredValue>& forked,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes) {
    auto g = method.graph();
    Node* fork_node;
    TypePtr out_type;

    fork_node = g->insertNode(method.graph()->create(prim::forkClosure, 1))
                    ->setSourceRange(loc);

    // We create a fork by emitting a closure and setting the closure output
    // into the fork input. If a closure doesn't already exist, we create one.
    {
      WithInsertPoint insert(fork_node);
      if (ClosureValue* sv = dynamic_cast<ClosureValue*>(forked.get())) {
        Value* closure_output = sv->asValue(loc, method);
        Block* closure_block = closure_output->node()->blocks().at(0);
        TORCH_INTERNAL_ASSERT(closure_block->outputs().size() == 1);
        out_type = closure_block->outputs().at(0)->type();
        fork_node->addInput(closure_output);
      } else {
        auto emit_closure_body = [&](Block* closure_block) {
          auto fn_sugared_output =
              forked->call(loc, method, inputs, attributes, 1);
          auto fn_simple_output = fn_sugared_output->asValue(loc, method);
          closure_block->registerOutput(fn_simple_output);
          out_type = fn_simple_output->type();
        };
        auto closure_value = emitClosure(emit_closure_body);
        fork_node->addInput(closure_value->asValue(loc, method));
      }
    }
    Value* node_output =
        fork_node->output()->setType(FutureType::create(out_type));
    return std::make_shared<SimpleValue>(node_output);
  }

  Value* emitSimpleExpr(
      const TreeRef& tree,
      const TypePtr& type_hint = nullptr) {
    switch (tree->kind()) {
      case TK_FLOOR_DIV:
      case '@': {
        const auto& inputs = tree->trees();
        auto kind = getNodeKind(tree->kind(), inputs.size());
        auto named_values = getNamedValues(inputs, /*maybe_unpack=*/false);
        return emitBuiltinCall(
            tree->range(),
            *method.graph(),
            kind,
            c10::nullopt,
            named_values,
            {},
            /*required=*/true);
      }
      case TK_IN:
      case TK_POW:
      case TK_NE:
      case TK_EQ:
      case '<':
      case '>':
      case TK_LE:
      case TK_GE:
      case '*':
      case '/':
      case '+':
      case '-':
      case '%':
      case '&':
      case '|':
      case '^': {
        const auto& inputs = tree->trees();
        auto kind = getNodeKind(tree->kind(), inputs.size());
        auto overload = getOperatorOverload(tree->kind(), inputs.size());
        auto named_values = getNamedValues(inputs, /*maybe_unpack=*/false);

        if (tree->kind() == TK_IN) {
          // For `in` the arguments are in reverse order (the object being
          // checked is second)
          std::iter_swap(named_values.begin() + 0, named_values.begin() + 1);
        }

        return asSimple(
            makeMagic(
                overload, std::make_shared<BuiltinFunction>(kind, at::nullopt))
                ->call(tree->range(), method, named_values, {}, 0));
      }
      case TK_IS:
      case TK_ISNOT:
      case TK_AND:
      case TK_OR:
      case TK_NOT: {
        return emitCondExpr(Expr(tree)).value();
      }
      case TK_UNARY_MINUS: {
        return emitUnaryOp(tree, "__neg__", aten::neg);
      }
      case '~': {
        return emitUnaryOp(tree, "__invert__", aten::bitwise_not);
      }
      case TK_STARRED: {
        throw ErrorReport(tree)
            << "Unexpected starred expansion. File a bug report";
      }
      case TK_CONST: {
        return emitConst(Const(tree));
      } break;
      case TK_TRUE: {
        return graph->insertConstant(true, tree->range());
      } break;
      case TK_FALSE: {
        return graph->insertConstant(false, tree->range());
      } break;
      case TK_NONE: {
        return graph->insertConstant(IValue(), tree->range());
      } break;
      case TK_SUBSCRIPT: {
        return emitSubscript(Subscript(tree));
      } break;
      case TK_IF_EXPR: {
        return emitTernaryIf(TernaryIf(tree));
      } break;
      case TK_STRINGLITERAL: {
        return emitStringLiteral(StringLiteral(tree));
      } break;
      case TK_LIST_LITERAL: {
        auto ll = ListLiteral(tree);
        auto values = getValues(ll.inputs(), /*maybe_unpack=*/true);

        // determine the element type of the list
        // if we have a type hint of List[T], use T
        // if the list is non-empty use type_of(list[0])
        // otherwise assume it is List[Tensor]
        TypePtr elem_type = TensorType::get();
        if (type_hint && type_hint->kind() == TypeKind::ListType) {
          elem_type = type_hint->expect<ListType>()->getElementType();
        } else if (!values.empty()) {
          elem_type = values.at(0)->type();
        }

        // Tensors are special because they have dymnamic properties. So any
        // list containing tensors should be typed with the unified typeof all
        // the elements.
        if (elem_type->isSubtypeOf(TensorType::get())) {
          for (const auto& value : values) {
            elem_type = unifyTypes(elem_type, value->type()).value();
          }
        }
        for (auto v : values) {
          std::stringstream ss;
          if (!v->type()->isSubtypeOfExt(elem_type, &ss)) {
            throw ErrorReport(tree)
                << "Lists must contain only a single type, expected: "
                << elem_type->python_str() << " but found "
                << v->type()->python_str() << " instead.\n"
                << ss.str();
          }
        }
        Value* result =
            graph->insertNode(graph->createList(elem_type, values))->output();
        return result;
      } break;
      case TK_TUPLE_LITERAL: {
        auto ll = TupleLiteral(tree);
        auto values = getValues(ll.inputs(), /*maybe_unpack=*/true);
        return graph->insertNode(graph->createTuple(values))->output();
      } break;
      case TK_DICT_LITERAL: {
        auto dl = DictLiteral(tree);
        auto key_trees = dl.key_inputs().tree()->trees();
        auto value_trees = dl.value_inputs().tree()->trees();
        AT_ASSERT(key_trees.size() == value_trees.size());
        std::vector<Value*> keys, values;
        for (size_t i = 0; i < key_trees.size(); ++i) {
          keys.push_back(emitExpr(Expr(key_trees[i])));
          values.push_back(emitExpr(Expr(value_trees[i])));
        }

        TypePtr key_type = nullptr;
        TypePtr value_type = nullptr;

        if (type_hint && type_hint->kind() == TypeKind::DictType) {
          auto dict_type = type_hint->expect<DictType>();
          key_type = dict_type->getKeyType();
          value_type = dict_type->getValueType();
        } else if (!keys.empty()) {
          key_type = keys.at(0)->type();
          value_type = values.at(0)->type();
        } else {
          key_type = StringType::get();
          value_type = TensorType::get();
        }
        AT_ASSERT(key_type != nullptr && value_type != nullptr);

        return graph
            ->insertNode(graph->createDict(key_type, value_type, keys, values))
            ->output();
      } break;
      case TK_LIST_COMP: {
        auto lc = ListComp(tree);
        return emitListComprehension(lc);
      } break;
      default:
        throw ErrorReport(tree) << "Cannot emit expr for: " << tree;
    }
  }

  Value* emitConst(const Const& c) {
    if (c.isFloatingPoint())
      return materializeConstant(
          c.asFloatingPoint(), *graph, c.range(), fp_constants);
    else
      return materializeConstant(
          c.asIntegral(), *graph, c.range(), integral_constants);
  }

  Value* emitStringLiteral(const StringLiteral& c) {
    return insertConstant(*graph, c.text(), c.range());
  }

  // Desugars select indexing: tensor[i] -> tensor.select(dim, i)
  Value* emitSelect(
      const SourceRange& loc,
      Value* input,
      Value* dim,
      Value* index) {
    return emitBuiltinCall(
        loc, *graph, aten::select, c10::nullopt, {input, dim, index}, {}, true);
  }

  // Desugars slice indexing: tensor[begin:end] -> tensor.slice(dim, begin, end,
  // 1)
  Value* emitSlice(
      const SourceRange& loc,
      Value* input,
      Value* dim, // Only used for tensor slicing
      const SliceExpr& slice) {
    std::vector<NamedValue> args;
    args.reserve(4);
    args.emplace_back(loc, "self", input);

    // XXX: If list slicing becomes more complicated or stops using
    // aten::slice, we should separate it from this function.
    if (dim) {
      AT_ASSERT(input->type()->isSubtypeOf(TensorType::get()));

      args.emplace_back(dim);
    } else {
      AT_ASSERT(!input->type()->isSubtypeOf(TensorType::get()));
    }

    args.emplace_back(loc, "begin", emitExpr(Expr(slice.startOr(0))));
    const auto has_end = slice.end().present();
    if (has_end) {
      args.emplace_back(loc, "end", emitExpr(Expr(slice.end().get())));
    }
    if (input->type()->cast<TupleType>()) {
      auto has_step = slice.step().present();
      if (has_step) {
        // TODO: add support for slicing tuples with a step
        throw ErrorReport(loc)
            << "Unsupported operation: slicing tuples with a step isn't supported";
      }

      if (has_end) {
        return emitTupleSlice(loc, args[0], args[1], /*end*/ args[2]);
      } else {
        return emitTupleSlice(loc, args[0], args[1], c10::nullopt);
      }
    }

    auto step = emitExpr(Expr(slice.stepOr(1)));
    NamedValue step_nv = NamedValue(loc, "step", step);
    return emitBuiltinCall(
        loc, *graph, aten::slice, c10::nullopt, args, {step_nv}, true);
  }

  Value* emitUnsqueeze(const SourceRange& loc, Value* input, Value* dim_val) {
    return emitBuiltinCall(
        loc, *graph, aten::unsqueeze, c10::nullopt, {input, dim_val}, {}, true);
  }

  Value* emitIndex(
      const SourceRange& loc,
      Value* input,
      at::ArrayRef<Value*> indices) {
    // NB: the index of aten::index should be a type of List[Optional[Tensor]],
    // this is to support the case like t[:, :, 1] where : here indicates a
    // None/undefined tensor(optional tensor)
    auto* index =
        graph->insertNode(graph->createList(OptionalType::ofTensor(), indices))
            ->output();
    return emitBuiltinCall(
        loc, *graph, aten::index, c10::nullopt, {input, index}, {}, true);
  }

  // Emits multidimensional slicing with int and slice indices.
  // Returns:
  // - Value*: the input after it has been indexed by int and slice indices.
  // - vector<Value*>: A list of tensor Value* indices that have not been
  // applied yet.
  //   Should be NULL at indices where sliceable (post-slicing) isn't indexed by
  //   a tensor.
  std::pair<Value*, std::vector<Value*>> emitIntAndSliceIndexing(
      const SourceRange& loc,
      Value* sliceable,
      const List<Expr>& subscript_exprs) {
    // Overall, to handle indexing (other than Tensors), we need to handle a
    // couple different things. For example, for x[1:3, None, 4], each of these
    // different index types (slice, None, and integer) result in different
    // number of dimensions. Slicing doesn't change the number of dimensions,
    // None adds a dimension, and integer removes a dimension. As these indexing
    // operations are applied left to right, the actual index that it's being
    // applied to depends on the previous operations. Ellipses indexing throws
    // another wrinkle. Ellipses selects any remaining unspecified dimensions.
    // Thus, for indexes following an ellipses, the actual index an indexing
    // operation is being applied to depends on the operations to the right.
    // Thus, we do two passes, one from left to right up until the ellipses, and
    // one from right to left.

    std::vector<Value*> tensor_indices;

    auto insert_value_for_dim = [&](int64_t dim) {
      return graph->insertConstant(dim, loc);
    };
    std::vector<int64_t> dims(subscript_exprs.size());
    std::vector<c10::optional<Value*>> exprs(
        subscript_exprs.size(), c10::nullopt);

    auto handle_indexing = [&](const Expr& subscript_expr,
                               int expr_idx,
                               int64_t dim,
                               bool is_reverse = false) {
      dims[expr_idx] = dim;
      if (subscript_expr.kind() == TK_SLICE_EXPR) {
        if (is_reverse) {
          return dim - 1;
        } else {
          return dim + 1;
        }
      }
      TypePtr type_hint = OptionalType::ofTensor();
      if (subscript_expr.kind() == TK_NONE) {
        type_hint = NoneType::get();
      }
      auto index = emitExpr(subscript_expr, type_hint);
      exprs[expr_idx] = index;
      if (index->type()->isSubtypeOf(NoneType::get())) {
        if (is_reverse) {
          return dim;
        } else {
          return dim + 1;
        }
      } else if (index->type() == IntType::get()) {
        if (is_reverse) {
          return dim - 1;
        } else {
          return dim;
        }
      } else if (index->type()->isSubtypeOf(OptionalType::ofTensor())) {
        if (is_reverse) {
          throw ErrorReport(loc)
              << "Ellipses followed by tensor indexing is currently not supported";
        } else {
          return dim + 1;
        }
      } else {
        throw ErrorReport(loc)
            << "Unsupported operation: indexing tensor with unsupported index type '"
            << index->type()->python_str()
            << "'. Only ints, slices, and tensors are supported";
      }
    };

    size_t idx = 0;
    int64_t dim = 0;
    for (; idx < subscript_exprs.size(); idx++) {
      auto subscript_expr = subscript_exprs[idx];
      if (subscript_expr.kind() == TK_DOTS) {
        break;
      }
      dim = handle_indexing(subscript_expr, idx, dim, /*is_reverse=*/false);
    }
    int64_t rdim = -1;
    for (size_t rev_idx = subscript_exprs.size() - 1; rev_idx > idx;
         rev_idx--) {
      auto subscript_expr = subscript_exprs[rev_idx];
      if (subscript_expr.kind() == TK_DOTS) {
        throw ErrorReport(loc)
            << "An index can only have a single ellipsis ('...')";
      }
      rdim =
          handle_indexing(subscript_expr, rev_idx, rdim, /*is_reverse=*/true);
    }
    for (size_t i = 0; i < exprs.size(); i++) {
      if (!exprs[i].has_value()) {
        if (subscript_exprs[i].kind() == TK_SLICE_EXPR) {
          sliceable = emitSlice(
              loc,
              sliceable,
              insert_value_for_dim(dims[i]),
              SliceExpr(subscript_exprs[i]));
        }
        continue;
      }
      auto expr = exprs[i].value();
      if (expr->type()->isSubtypeOf(NoneType::get())) {
        sliceable =
            emitUnsqueeze(loc, sliceable, insert_value_for_dim(dims[i]));
      } else if (expr->type() == IntType::get()) {
        sliceable =
            emitSelect(loc, sliceable, insert_value_for_dim(dims[i]), expr);
      } else if (expr->type()->isSubtypeOf(OptionalType::ofTensor())) {
        tensor_indices.resize(dims[i] + 1);
        tensor_indices[dims[i]] = expr;
      } else {
        TORCH_INTERNAL_ASSERT(
            "Trying to process index type that we don't support.");
      }
    }
    // at::index takes in a List[Optional[Tensor]] where some dims can be None.
    // create None node with optional tensor output type and pass to at::index.
    for (auto& index : tensor_indices) {
      if (index == nullptr) {
        index = graph->insertNode(graph->createNone())->output();
      }
    }
    return std::make_pair(sliceable, tensor_indices);
  }

  // Desugars multidim slicing into slice/select/index/unsqueeze calls.
  //
  // XXX: Errors in user code are not elegantly reported.
  // Let's say someone were to do the following:
  //   @torch.jit.script
  //   def fn(x):
  //       return x[0, 1]
  //   fn(torch.randn(5))
  // Because we desugar this into two aten::select ops, the error message
  // complains about aten::select failing rather than there "not being
  // enough dimensions to index".
  //
  // The strategy is to slice and select the tensor for int and slices first
  // in one pass and then apply at::index on the result of the
  // slicing/selecting. Call the tensor after we've applied slice / select the
  // `sliced`. tensor_indices should have the same size as sliced.dim():
  // - tensor_indices[i] = NULL if we should not index `sliced` at dim i
  // - tensor_indices[i] = t if we should index `sliced` at dim i with tensor t.
  Value* emitMultidimSlicing(
      const SourceRange& loc,
      Value* sliceable,
      const List<Expr>& subscript_exprs) {
    if (!sliceable->type()->isSubtypeOf(TensorType::get())) {
      throw ErrorReport(loc)
          << "Unsupported operation: attempted to use multidimensional "
          << "indexing on a non-tensor type";
    }

    std::vector<Value*> tensor_indices;
    std::tie(sliceable, tensor_indices) =
        emitIntAndSliceIndexing(loc, sliceable, subscript_exprs);

    if (tensor_indices.empty()) {
      // XXX: Might need to at::alias this when we support mutability
      return sliceable;
    }

    return emitIndex(loc, sliceable, tensor_indices);
  }

  // Desugars slice syntactic sugar tensor[begin:end] -> tensor.slice(begin,
  // end).
  Value* emitBasicSlice(
      const SourceRange& loc,
      Value* sliceable,
      const List<Expr>& subscript_exprs) {
    AT_ASSERT(subscript_exprs.size() == 1);
    AT_ASSERT(subscript_exprs[0].kind() == TK_SLICE_EXPR);
    auto slice_exp = SliceExpr(subscript_exprs[0]);
    Value* maybe_dim = nullptr;
    if (sliceable->type()->isSubtypeOf(TensorType::get())) {
      // If the sliceable object is a tensor, specify a default dimension
      maybe_dim = graph->insertConstant(0, loc);
    }
    return emitSlice(loc, sliceable, maybe_dim, slice_exp);
  }

  int64_t getAdjTupleIndex(
      const SourceRange& loc,
      const TupleTypePtr& tuple_type,
      int64_t input_index,
      bool allow_out_of_bounds) {
    // set index to be positive to simplify logic in runtime
    int64_t adj_index = input_index;
    int64_t tuple_len = tuple_type->elements().size();
    if (input_index < 0) {
      adj_index = tuple_len + input_index;
    }
    if (!allow_out_of_bounds && (adj_index >= tuple_len || adj_index < 0)) {
      throw ErrorReport(loc) << "Tuple index out of range. Tuple is length "
                             << tuple_len << " and index is " << input_index;
    }
    return adj_index;
  }

  // When a list is marked const in a module, it gets converted to a tuple.
  // The result is indexing into a Tuple which contains only one type
  // is quite common. since indexing will likely be done in a for loop,
  // we do not want to invoke the overhead of converting the tuple to a list
  // each iter.
  Value* emitTupleIndex(
      const SourceRange& loc,
      Value* tuple_val,
      Value* idx_val) {
    auto tuple_typ = tuple_val->type()->cast<TupleType>();
    auto elems = tuple_typ->elements();
    TypePtr output_type;
    if (idx_val->type() != IntType::get()) {
      throw ErrorReport(loc) << "tuple index must be an integer";
    }
    auto idx = toIValue(idx_val);
    if (!idx) {
      if (elems.size() == 0 ||
          !convertibleToList(tuple_typ, ListType::create(elems[0]))) {
        throw ErrorReport(loc)
            << "Cannot index into a " << tuple_typ->python_str()
            << " with a non-integer literal because we cannot resolve the output type";
      }
      output_type = elems[0];
    } else {
      auto adj_index = getAdjTupleIndex(
          loc, tuple_typ, idx->toInt(), /*allow_out_of_bounds*/ false);
      output_type = elems[adj_index];
    }
    return graph
        ->insertNode(graph->createTupleIndex(tuple_val, idx_val, output_type))
        ->output();
  }

  int64_t getSliceInd(Value* idx_val, const SourceRange& loc) {
    auto ivalue = toIValue(idx_val);
    if (ivalue && ivalue->isInt()) {
      return ivalue->to<int64_t>();
    } else {
      throw ErrorReport(loc) << "tuple slice indices must be integer constants";
    }
  }

  Value* emitTupleSlice(
      const SourceRange& loc,
      const NamedValue& tuple_val,
      const NamedValue& beg_val,
      const at::optional<NamedValue>& end_val) {
    auto tuple_type = tuple_val.value(*graph)->type()->expect<TupleType>();
    int64_t beg = getAdjTupleIndex(
        loc,
        tuple_type,
        getSliceInd(beg_val.value(*graph), loc),
        /*allow_out_of_bounds*/ true);
    int64_t end;
    int64_t tuple_len = tuple_type->elements().size();
    if (end_val) {
      end = getAdjTupleIndex(
          loc, tuple_type, getSliceInd(end_val->value(*graph), loc), true);
    } else {
      end = tuple_len;
    }
    // slicing does not throw out of bounds errors
    end = std::min(std::max((int64_t)0, end), tuple_len);
    beg = std::min(std::max((int64_t)0, beg), tuple_len);

    return graph
        ->insertNode(graph->createTupleSlice(tuple_val.value(*graph), beg, end))
        ->output();
  }

  Value* emitSubscript(const Subscript& subscript) {
    const SugaredValuePtr sv = emitSugaredExpr(subscript.value(), 1);
    const List<Expr>& subscript_exprs = subscript.subscript_exprs();
    const SourceRange& range = subscript.range();
    const SourceRange& val_range = subscript.value().range();
    if (subscript_exprs.size() != 1) {
      return emitMultidimSlicing(
          range, sv->asValue(val_range, method), subscript_exprs);
    }
    if (subscript_exprs[0].kind() == TK_SLICE_EXPR) {
      return emitBasicSlice(
          range, sv->asValue(val_range, method), subscript_exprs);
    } else {
      // Desugars gather syntactic sugar foo[i]
      Value* idx = emitExpr(subscript_exprs[0]);
      Value* val = sv->asValue(val_range, method);
      AT_ASSERT(subscript_exprs.size() == 1);

      if (val->type()->cast<TupleType>()) {
        return emitTupleIndex(range, sv->asValue(val_range, method), idx);
      } else if (val->type()->isSubtypeOf(TensorType::get())) {
        return emitMultidimSlicing(range, val, subscript_exprs);
      } else {
        return sv->getitem(range, method, idx);
      }
    }
  }
};

struct FunctionResolver : public Resolver {
  explicit FunctionResolver(
      Resolver* otherResolver,
      const std::unordered_map<std::string, Function*>& functionTable)
      : otherResolver_(otherResolver), functionTable_(functionTable) {}

  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      Function& m,
      const SourceRange& loc) override {
    auto it = functionTable_.find(name);
    if (it != functionTable_.end()) {
      return std::make_shared<FunctionValue>(it->second);
    }
    return otherResolver_->resolveValue(name, m, loc);
  }

  TypePtr resolveType(const std::string& name, const SourceRange& loc)
      override {
    return otherResolver_->resolveType(name, loc);
  }

 private:
  Resolver* otherResolver_;
  const std::unordered_map<std::string, Function*>& functionTable_;
};

CompilationUnit::CompilationUnit(const std::string& source)
    : CompilationUnit() {
  // calles the define with native resolver to generate the graph for functions
  define(c10::nullopt, source, nativeResolver(), nullptr);
}

c10::QualifiedName CompilationUnit::mangle(
    const c10::QualifiedName& name) const {
  static const std::string manglePrefix = "___torch_mangle_";
  std::vector<std::string> atoms = name.atoms();

  // Search for an already-existing mangle namespace.
  // If the name is already mangled, just bump the integer.
  for (auto& atom : atoms) {
    auto pos = atom.find(manglePrefix);
    if (pos != std::string::npos) {
      std::string newAtom;
      newAtom.reserve(atom.size());
      // Append the part of the name up to the end of the prefix
      newAtom.append(atom, 0, pos);
      newAtom.append(std::to_string(mangleIndex_++));
      atom = newAtom;
      return QualifiedName(atoms);
    }
  }

  // Otherwise add a mangle namespace right before the basename
  TORCH_INTERNAL_ASSERT(!atoms.empty());
  atoms.insert(atoms.end() - 1, manglePrefix + std::to_string(mangleIndex_++));
  return QualifiedName(atoms);
}

std::unique_ptr<Function> CompilationUnit::define(
    const c10::optional<QualifiedName>& prefix,
    const Def& def,
    const ResolverPtr& resolver,
    const Self* self,
    const std::unordered_map<std::string, Function*>& function_table,
    bool shouldMangle) const {
  TORCH_INTERNAL_ASSERT(resolver);
  auto _resolver = resolver;
  if (!self) {
    // if self is defined, then these are methods and do not go into the
    // global namespace otherwise, they get defined together so we add them to
    // the function table so the methods can see each other
    _resolver =
        std::make_shared<FunctionResolver>(resolver.get(), function_table);
  }
  auto creator = [def, _resolver, self](Function& method) {
    // Store the function name so that it can be referenced if there is an error
    // while compiling this function
    ErrorReport::CallStack call(
        self ? method.qualname().qualifiedName() : method.qualname().name());
    to_ir(def, _resolver, self, method);
  };
  auto name = prefix ? QualifiedName(*prefix, def.name().name())
                     : QualifiedName(def.name().name());
  if (shouldMangle) {
    // If `shouldMangle` is set, we should generate a unique name for this
    // function if there is already an existing one.
    if (auto fn = find_function(name)) {
      name = mangle(name);
    }
  }
  auto fn = torch::make_unique<Function>(
      std::move(name), std::make_shared<Graph>(), creator);
  if (self) {
    // Register this as a method on `self`'s type
    self->getClassType()->addMethod(fn.get());
  }
  return fn;
}

std::vector<Function*> CompilationUnit::define(
    const c10::optional<QualifiedName>& prefix,
    const std::vector<Def>& definitions,
    const std::vector<ResolverPtr>& resolvers,
    const Self* self,
    bool shouldMangle) {
  TORCH_INTERNAL_ASSERT(definitions.size() == resolvers.size());
  std::vector<Function*> functions;
  std::unordered_map<std::string, Function*> function_table;

  for (size_t i = 0; i < definitions.size(); i++) {
    auto fn = define(
        prefix,
        definitions[i],
        resolvers[i],
        self,
        function_table,
        shouldMangle);
    const auto& name = fn->name();
    function_table[name] = fn.get();
    functions.push_back(fn.get());
    register_function(std::move(fn));
  }

  // We need to compile `__init__` first, since it can determine what attributes
  // are available to other methods. So reorder the definitions accordingly.
  for (size_t i = 0; i < definitions.size(); i++) {
    const auto& def = definitions[i];
    if (def.name().name() == "__init__") {
      functions[i]->ensure_defined();
    }
  }

  for (Function* function : functions) {
    function->ensure_defined();
  }
  return functions;
}

std::vector<Function*> CompilationUnit::define(
    const c10::optional<QualifiedName>& prefix,
    const std::string& source,
    const ResolverPtr& resolver,
    const Self* self) {
  Parser p(std::make_shared<Source>(source, "<string>", 1));
  std::vector<Def> definitions;
  std::vector<ResolverPtr> resolvers;
  while (p.lexer().cur().kind != TK_EOF) {
    auto def = Def(p.parseFunction(/*is_method=*/bool(self)));
    definitions.push_back(def);
    resolvers.push_back(resolver);
  }
  return define(prefix, definitions, resolvers, self);
}

void runCleanupPasses(std::shared_ptr<Graph>& to_clean, bool convert_ssa) {
  // the graph including closures is converted to ssa in the first pass,
  // so subsequent cleanups do not need reconvert it
  if (convert_ssa) {
    ConvertToSSA(to_clean);
    // convert loops with an iter and body condition specified to
    // python-recognize while loops. we do this so they can be exported,
    // and run the pass early to avoid jitter. Like conversion to SSA,
    // it only needs to run once.
    CanonicalizeModifiedLoops(to_clean);
  }
  // NB ORDERING: SSA conversion has to occur before
  // lifting of closures and forks, this way closures are converted
  // to SSA while part of their original graph, and closures are ready to
  // be inlined into forked closures
  liftClosures(to_clean);
  inlineForkedClosures(to_clean);
  if (script::getInlineEverythingMode()) {
    Inline(*to_clean);
  }
  // remove any uses of tuples that we inserted that are not needed
  LowerSimpleTuples(to_clean);
  ConstantPooling(to_clean);
  // For jitter
  CanonicalizeOutputs(to_clean);
}

// we consider _N where N is a number, to be a non-meaningful name
// and do not record it as a unique name. This allows python printing to
// be able to export and import more consistently named graphs
bool meaningfulName(const std::string& name) {
  if (name.size() == 0)
    return false;
  if (name[0] == '$')
    return false;
  if (name[0] != '_')
    return true;
  for (size_t i = 1; i < name.size(); ++i) {
    if (!isdigit(name[i]))
      return true;
  }
  return false;
}

void lambdaLiftFork(Node* fork_node) {
  // Fork a new graph from its orignal owning graph
  auto forked_graph = std::make_shared<Graph>();
  auto body_block = fork_node->blocks()[0];

  // Make sure we capture everything in the new graph.
  // The uncaptured values will be added to the fork signature.
  std::unordered_map<Value*, Value*> uncaptures_map;
  auto env = [&](Value* v) -> Value* {
    if (!uncaptures_map.count(v)) {
      // Capture values for both graphs
      uncaptures_map[v] = forked_graph->addInput()->copyMetadata(v);
      fork_node->addInput(v);
    }
    return uncaptures_map[v];
  };
  forked_graph->block()->cloneFrom(body_block, env);

  // Separate the subgraph and clean up the orignal one
  fork_node->g_(attr::Subgraph, forked_graph);
  fork_node->eraseBlock(0);
}

void CompilationUnit::define_interface(
    const c10::QualifiedName& qualifiedName,
    const ClassDef& classDef,
    ResolverPtr rcb) {
  ScriptTypeParser typeParser(rcb);
  InterfaceTypePtr iface =
      InterfaceType::create(c10::QualifiedName(qualifiedName));
  for (const Stmt& stmt : classDef.body()) {
    if (stmt.kind() != TK_DEF) {
      throw ErrorReport(stmt)
          << "interface declartions can only contain method definitions";
    }
    auto method_def = Def(stmt);
    if (!method_def.decl().return_type().present()) {
      throw ErrorReport(method_def)
          << "interface declarations must have a return type annotated.";
    }
    FunctionSchema schema =
        typeParser.parseSchemaFromDef(method_def, /* skip_self*/ true);
    // need to add self as the first because we skipped it
    std::vector<Argument> arguments;
    arguments.emplace_back(
        Argument(method_def.decl().params()[0].ident().name(), iface));
    arguments.insert(
        arguments.end(), schema.arguments().begin(), schema.arguments().end());
    iface->addMethod(schema.cloneWithArguments(std::move(arguments)));
    if (method_def.statements().size() != 1 ||
        method_def.statements()[0].kind() != TK_PASS) {
      throw ErrorReport(method_def.range())
          << "interfaces declarations should only contain a single 'pass' statement.";
    }
  }
  this->register_type(iface);
}

} // namespace script
} // namespace jit
} // namespace torch
