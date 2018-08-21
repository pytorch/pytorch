#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/passes/lower_tuples.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/script/parser.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/jit/operator.h"

#include "torch/csrc/jit/constants.h"

#include "ATen/core/optional.h"


#include <climits>
#include <set>

namespace torch {
namespace jit {
namespace script {

using SugaredValuePtr = std::shared_ptr<SugaredValue>;
using FunctionTable = std::unordered_map<std::string, Method&>;
using ValueTable = std::unordered_map<std::string, SugaredValuePtr>;
using AttributeMap = std::unordered_map<std::string, Const>;
using ListAttributeMap = std::unordered_map<std::string, std::vector<Const>>;

struct NoneValue : SugaredValue {
  NoneValue() {}
  virtual std::string kind() const override {
    return "None";
  }
};

struct PrintValue : public SugaredValue {
  std::string kind() const override {
    return "print";
  }
  std::shared_ptr<SugaredValue> call(
    SourceRange loc,
    Method & m,
    at::ArrayRef<NamedValue> inputs,
    at::ArrayRef<NamedValue> attributes,
    size_t n_binders) override {
      auto& g = *m.graph();
      if (!attributes.empty())
        throw ErrorReport(loc) << "print doesn't accept any keyword arguments";

      //temporary hack to allow print statements to work in python 2, where
      //print(a, b) is treated as a (a, b) tuple input.

      std::vector<Value*> lowered_inputs = toValues(*m.graph(), inputs);
      if(lowered_inputs.size() == 1 && lowered_inputs.at(0)->node()->kind() == prim::TupleConstruct) {
        auto input = lowered_inputs[0];
        for(size_t j = 0; j < input->node()->inputs().size(); ++j) {
          lowered_inputs.insert(lowered_inputs.begin() + 1 + j, input->node()->inputs().at(j));
        }
        lowered_inputs.erase(lowered_inputs.begin());
      }
      g.insertNode(g.create(prim::Print, lowered_inputs, 0)
                       ->setSourceLocation(std::make_shared<SourceRange>(loc)));
      return std::make_shared<NoneValue>();
  }
};

static Value* typeCast(const SourceRange& loc, Value* value, TypePtr dst) {
  auto& graph = *value->owningGraph();
  const TypePtr orig = value->type();
  Node* n = nullptr;

  if(dst->isSubtypeOf(DynamicType::get()) && orig->isSubtypeOf(NumberType::get())) {
    n = graph.createNumToTensor(value);
  } else if (dst->isSubtypeOf(NumberType::get()) && orig->isSubtypeOf(DynamicType::get())) {
    n = graph.createTensorToNum(dst, value);
  } else if(dst->isSubtypeOf(IntType::get()) && orig->isSubtypeOf(FloatType::get())) {
    n = graph.createFloatToInt(value);
  } else if(dst->isSubtypeOf(FloatType::get()) && orig->isSubtypeOf(IntType::get())) {
    n = graph.createIntToFloat(value);
  } else {
    throw ErrorReport(loc) << "Cannot cast type '" << orig->str() << "' to type '"
      << dst->str() << "'.";
  }

  auto* result = graph.insertNode(n)
      ->setSourceLocation(std::make_shared<SourceRange>(loc))
      ->output();
  return result;
}

// expressions like int(x)
struct CastValue : public SugaredValue {
  CastValue(TypePtr type)
  : type(type) {}
  std::string kind() const override {
    std::stringstream ss;
    ss << "<" << type->str() << " cast primitive>";
    return ss.str();
  }
  std::shared_ptr<SugaredValue> call(
    SourceRange loc,
    Method & m,
    at::ArrayRef<NamedValue> inputs,
    at::ArrayRef<NamedValue> attributes,
    size_t n_binders) override {
      if (!attributes.empty())
        throw ErrorReport(loc) << "casts do not accept any keyword arguments";
      if (inputs.size() != 1)
        throw ErrorReport(loc) << "expected a single argument for cast";
      auto values = toValues(*m.graph(), inputs);
      Value* input = values.at(0);
      if(!input->type()->isSubtypeOf(type)) {
          input = typeCast(loc, input, type);
      }
      return std::make_shared<SimpleValue>(input);
  }
private:
  TypePtr type;
};

// Auxiliary data structure for desugaring variable binding into our always
// explicitly scoped language as we descend down
// nested control structures in the frontend (which themselves don't introduce
// scopes)
//
// The algorithm is roughly as follows:
// 1) While emitting a block within a control operator, add inputs and outputs
//      from the block for each value referenced (both "reads" and "writes").
//      This sets the value up as a candidate loop carried dependency.
// 2) When we reach the end of the block, examine all the values in the current
//      scope's value map. If the name also resides in an outer scope with a
//      different Value*, this is a true loop-carried dependency. If not, this
//      value was not assigned to. Replace all references to the block input
//      with the Value* pointed to in the tightest enclosing scope. Then delete
//      that block input and output.
// 3) When we emit the actual control operator, take all of the loop-carried
//      dependency values as inputs and return them as outputs from the control
//      op
//
//  Note that an alternative implementation could only add the loop-carried dep
//      inputs and outputs when we see a value that is mutated. This, however
//      requires replacing all references to that value *within the current
//      block* with a new input. That is to say: we need to traverse the pre-
//      decessor nodes and replace inputs that reference that value with the
//      newly-created input. This could be made less expensive with a change to
//      the IR API, but for now we choose to pessimisitically create inputs and
//      delete unnecessary ones later with replaceAllusesWith().
struct Environment {
  Environment(Method & method, const Resolver& resolver, Block* b, std::shared_ptr<Environment> next = nullptr)
      : method(method), resolver(resolver), b(b), next(next) {}

  Method & method;
  const Resolver& resolver;
  std::vector<std::string> captured_inputs;
  Block* b;

  std::shared_ptr<Environment> next;

  SugaredValuePtr findInThisFrame(const std::string& name) {
    auto it = value_table.find(name);
    if (it != value_table.end()) {
      return it->second;
    }
    return nullptr;
  }

  SugaredValuePtr findInParentFrame(const std::string& name) {
    return next ? next->findInAnyFrame(name) : nullptr;
  }

  SugaredValuePtr findInAnyFrame(const std::string& name) {
    for (auto runner = this; runner; runner = runner->next.get()) {
      if(auto r = runner->findInThisFrame(name)) {
        return r;
      }
    }
    return nullptr;
  }

  Value* getValueInThisFrame(const SourceRange& loc, const std::string& name) {
    return value_table.at(name)->asValue(loc, method);
  }

  SugaredValuePtr createCapturedInput(Value* orig, const std::string& name) {
    // Create the input
    Value* new_input = b->addInput()->setType(orig->type());

    // Associate this name with this value
    auto sv = std::make_shared<SimpleValue>(new_input);
    value_table[name] = sv;

    // List as a positional input
    captured_inputs.push_back(name);

    return sv;
  }

  SugaredValuePtr createCapturedInputIfNeeded(const SourceRange& loc, std::string ident) {
    auto in_frame = findInThisFrame(ident);
    if (in_frame) {
      return in_frame;
    }

    // recursively handles the case where parent blocks are also loops
    auto from_parent = next ? next->createCapturedInputIfNeeded(loc, ident) : nullptr;

    // recursively create the captured input if it is the loop block
    if (from_parent && getBlockOwningKind() == prim::Loop) {
      if (Value* simple_val = asSimple(from_parent))
        from_parent = createCapturedInput(simple_val, ident);
    }
    return from_parent;
  }

  Block* block() {
    return b;
  }
  Symbol getBlockOwningKind() {
    Symbol owning_kind = Symbol();
    if (b->owningNode()) {
      owning_kind = b->owningNode()->kind();
    }
    return owning_kind;
  }

  void setVar(const SourceRange& loc, const std::string& name, Value* value) {
    setSugaredVar(loc, name, std::make_shared<SimpleValue>(value));
  }
  static Value* asSimple(SugaredValuePtr value) {
    if(SimpleValue* sv = dynamic_cast<SimpleValue*>(value.get())) {
      return sv->getValue();
    }
    return nullptr;
  }

  void setSugaredVar(const SourceRange& loc, const std::string& name, SugaredValuePtr value) {
    Value* as_simple_value = asSimple(value);
    if (as_simple_value)
      as_simple_value->setUniqueName(name);
    // prevent re-assignment involving any sugared values
    // any reassignment like:
    // a = ...
    // while ...
    //   a = ..
    // requires 'a' to be first-class in the graph since its value depends on
    // control flow
    if(auto parent = findInParentFrame(name)) {
      if(!as_simple_value) {
        throw ErrorReport(loc) << "Cannot re-assign '" << name << "' to a value of type " << value->kind() <<
	" because " << name << " is not a first-class value.  Only reassignments to first-class values are allowed";
      }
      Value* simple_parent = asSimple(parent);
      if(!simple_parent) {
        throw ErrorReport(loc) << "Cannot re-assign '" << name << "' because it has type " << value->kind() <<
	" and " << name << " is not a first-class value.  Only reassignments to first-class values are allowed";
      }
      if (!as_simple_value->type()->isSubtypeOf(
              unshapedType(simple_parent->type()))) {
        std::stringstream errMsg;
        errMsg << "variable '" << name << "' previously has type "
               << simple_parent->type()->str()
               << " but is now being assigned to a value of type "
               << as_simple_value->type()->str();
        // Special-cased error msg if we're trying to assign to a tensor list.
        if (simple_parent->type()->kind() == TypeKind::ListType &&
            as_simple_value->type()->kind() == TypeKind::ListType) {
          errMsg << "\n. (Note: empty lists are constructed as Tensor[]; "
                 << "if you want an empty list of a different type, "
                 << "use `_construct_empty_foo_list`, "
                 << "where `foo` is `int` or `float`)";
        }
        throw ErrorReport(loc) << errMsg.str();
      }
    }
    if (as_simple_value)
      createCapturedInputIfNeeded(loc, name);
    value_table[name] = std::move(value);
  }

  SugaredValuePtr getSugaredVar(const Ident& ident, bool required=true) {
    return getSugaredVar(ident.name(), ident.range());
  }
  Value* getVar(const Ident& ident) {
    return getSugaredVar(ident)->asValue(ident.range(), method);
  }

  SugaredValuePtr getSugaredVar(const std::string& ident, SourceRange range, bool required=true) {
    auto retval = createCapturedInputIfNeeded(range, ident);

    if(!retval) {
      retval = resolver(ident, method, range);
    }

    if(!retval) {
      static std::unordered_map<std::string, SugaredValuePtr> globals = {
        {"print", std::make_shared<PrintValue>()},
        {"float", std::make_shared<CastValue>(FloatType::get())},
        {"int", std::make_shared<CastValue>(IntType::get())},
        {"bool", std::make_shared<CastValue>(IntType::get())},
        // todo(zach): remove when we can correctly export torch.full via ONNX
        // or we have implicit conversion that can convert numbers to tensors
        {"_to_tensor", std::make_shared<CastValue>(DynamicType::get()) },
      };
      auto it = globals.find(ident);
      if(it != globals.end())
        retval = it->second;
    }

    if (!retval && required) {
      throw ErrorReport(range) << "undefined value " << ident;
    }
    return retval;
  }

  Value* getVar(const std::string& ident, SourceRange range) {
    return getSugaredVar(ident, range)->asValue(range, method);
  }

  // Given that after emitting statements in a block, we've added block inputs
  // for all value references and assignments, delete inputs for which there was
  // no assignment, only references.
  void deleteExtraInputs(const SourceRange& loc) {
    // note: skip i == 0, it is the loop trip count for inputs
    // and the loop condition for outputs.
    // captured_inputs is indexed by i - 1 since it only contains loop
    // carried dependencies
    //          inputs: loop_counter, lcd0, lcd1, ...
    //         outputs: loop_condition, lcd0, lcd1, ...
    // captured_inputs: lcd0, lcd1, ...
    JIT_ASSERT(b->inputs().size() == b->outputs().size());
    JIT_ASSERT(b->inputs().size() == captured_inputs.size() + 1);
    for(size_t i = b->inputs().size() - 1; i > 0; i--) {
      // nothing changed along this loop
      if(b->inputs()[i] == b->outputs()[i]) {
        auto name = captured_inputs[i - 1];
        Value* orig = findInParentFrame(name)->asValue(loc, method);
        b->inputs()[i]->replaceAllUsesWith(orig);
        b->eraseInput(i);
        b->eraseOutput(i);
        captured_inputs.erase(captured_inputs.begin() + i - 1);
      }
    }
  }
  std::vector<std::string> definedVariables() {
    std::vector<std::string> result;
    for(auto & kv : value_table) {
      result.push_back(kv.first);
    }
    return result;
  }
private:
  ValueTable value_table;
};

Value* packOutputs(Graph& g, at::ArrayRef<Value*> values) {
  if(values.size() == 1) {
    return values[0];
  }
  return g.insertNode(g.createTuple(values))->output();
}

Value* createNumber(Graph& g, const SourceRange& loc, const at::Tensor& val) {
  JIT_ASSERT(val.numel() == 1);
  auto* output = g.insertConstant(val, loc);
  if (val.type().scalarType() == at::kLong) {
    output->setType(IntType::get());
  } else if (val.type().scalarType() == at::kFloat) {
    output->setType(FloatType::get());
  } else {
    throw ErrorReport(loc) << "createNumber with unknown scalar type ("
							<< val.type().scalarType() << "). Please file a bug report.";
  }
  return output;
}

at::optional<std::vector<int64_t>> getIntListAttribute(at::optional<int32_t> N, Value* input) {
  auto list = constant_as<Shared<jit::IntList>>(input);
  if(list)
    return list.value()->elements();

  // broadcast IntList[3] with value 4 -> {4, 4, 4}
  if(!N)
    return at::nullopt;

  auto r = constant_as<int64_t>(input);
  if(!r)
    return at::nullopt;

  // broadcast to attribute size
  return std::vector<int64_t>(*N, *r);
}

at::ArrayRef<Value*> createTupleUnpack(Value* v) {
  // small peephole optimization to ensure IntList attributes can still turn
  // into constants e.g. in x.expand([3, 4])
  if(v->node()->kind() == prim::TupleConstruct)
    return v->node()->inputs();
  auto & g = *v->owningGraph();
  return g.insertNode(g.createTupleUnpack(v))->outputs();
}

static inline bool isIntUsedAsIntList(
    const Value* value,
    const Argument& arg) {
  // Look for int[N]
  return value->type()->kind() == TypeKind::IntType &&
         *arg.type == *ListType::ofInts() && arg.N;
}

at::optional<std::vector<Value*>> tryMatchSchema(
  const FunctionSchema& schema,
  const SourceRange& loc,
  Graph& graph,
  at::ArrayRef<NamedValue> inputs,
  at::ArrayRef<NamedValue> attributes,
  std::ostream& failure_messages) {
    auto err = [&]() -> std::ostream& {
      failure_messages << "\nfor operator " << schema << ":\n";
      return failure_messages;
    };

    std::vector<at::optional<NamedValue>> positional_inputs(schema.arguments.size(), at::nullopt);

    size_t total_inputs = attributes.size() + inputs.size();
    if(total_inputs > schema.arguments.size()) {
      err() << "expected at most " << schema.arguments.size() << " arguments "
      << "but found " << total_inputs << "\n" << loc << "\n";
      return at::nullopt;
    }
    // fill in positional arguments
    for(size_t i = 0; i < inputs.size(); ++i) {
      positional_inputs[i] = inputs[i];
    }
    // fill in named arguments
    for(const NamedValue& nv : attributes) {
      auto idx = schema.argumentIndexWithName(nv.name());
      if(!idx) {
        err() << "unknown keyword argument '" << nv.name() << "'\n" << nv.locOr(loc);
        return at::nullopt;
      }
      if(positional_inputs[*idx]) {
        err() << "argument " <<  nv.name() << " specified twice \n" << nv.locOr(loc);
        return at::nullopt;
      }
      positional_inputs[*idx] = nv;
    }
    // fill in default values
    for(size_t i = 0; i < positional_inputs.size(); ++i) {
      if(positional_inputs[i])
        continue;
      auto default_value = schema.arguments[i].default_value;
      if(!default_value) {
        err() << "argument " << schema.arguments[i].name << " not provided.\n" << loc;
        return at::nullopt;
      }
      positional_inputs[i] = NamedValue(*default_value);
    }

    // check input types
    std::vector<Value*> matched_inputs;
    for(size_t i = 0; i < schema.arguments.size(); ++i) {
      Value* value = positional_inputs[i]->value(graph);
      const auto& arg = schema.arguments[i];

      // some functions that take lists of integers for fixed size arrays
      // also allow single ints to be passed in their place.
      // the single int is then repeated to the length of the list
      if (isIntUsedAsIntList(value, arg)) {
        std::vector<Value*> repeated(*arg.N, value);
        value = graph.insertNode(graph.createList(IntType::get(), repeated))->output();
      }

      // Allow homogeneous tuples to be casted implicitly to lists of appropriate types
      if (arg.type->kind() == TypeKind::ListType &&
          value->type()->kind() == TypeKind::TupleType &&
          value->type()->isSubtypeOf(arg.type)) {
        auto unpacked = createTupleUnpack(value);
        auto elem_type = arg.type->expect<ListType>()->getElementType();
        value = graph.insertNode(graph.createList(elem_type, unpacked))->output();
      }

      if (value->node()->kind() == prim::None){
        if (arg.type->isSubtypeOf(NumberType::get()))
          value = graph.insertConstant(at::Scalar(NAN), loc);
        else
          value = graph.insertNode(graph.createUndefined())->output();
      }

      if(!value->type()->isSubtypeOf(arg.type)) {
        err() << "expected a value of type " << arg.type->str() << " for argument '" << arg.name << "' but found "
              << value->type()->str() << "\n"
              << positional_inputs[i]->locOr(loc);
        return at::nullopt;
      }

      matched_inputs.push_back(value);
    }

    return matched_inputs;
}


static Value* tryEmitBuiltin(
  const std::shared_ptr<Operator>& op,
  std::stringstream& failure_messages,
  const SourceRange& loc,
  Graph& graph,
  Symbol name,
  at::ArrayRef<NamedValue> inputs,
  at::ArrayRef<NamedValue> attributes) {

  auto matched_inputs = tryMatchSchema(op->schema(), loc, graph, inputs, attributes, failure_messages);
  if(!matched_inputs)
    return nullptr;
  // we successfully matched this schema, construct the node

  auto n = graph.insertNode(graph.create(name, *matched_inputs, 0))
                ->setSourceLocation(std::make_shared<SourceRange>(loc));

  // special case for chunk when the chunks=<const> is known
  // DO NOT ADD MORE SPECIAL CASES HERE, REFACTOR INTO A FUNCTION IF
  // NEEDED
  if(n->kind() == aten::chunk) {
    auto value = constant_as<int64_t>((*matched_inputs)[1]);
    if(!value) {
      throw ErrorReport(loc) << "argument 'chunks' must be a constant";
    }
    for(int64_t i = 0; i < *value; ++i)
      n->addOutput();
  } else {
    for(auto & ret : op->schema().returns) {
      n->addOutput()->setType(ret.type);
    }
  }

  // assert that we did indeed create an op that has implementation
  // otherwise schema and dispatch are not in sync
  getOperation(n);

  return packOutputs(graph, n->outputs());
}

static std::string prefixLine(const std::string& str, std::string prefix) {
  std::stringstream ss;
  bool was_newline = true;
  for(auto c : str) {
    if(was_newline)
      ss << prefix;
    ss.put(c);
    was_newline = c == '\n';
  }
  return ss.str();
}

Value* emitBuiltinCall(
  const SourceRange& loc,
  Graph& graph,
  Symbol name,
  at::ArrayRef<NamedValue> inputs,
  at::ArrayRef<NamedValue> attributes,
  // if true, emitBuiltinCall will throw an exception if this builtin does not exist,
  // otherwise it will return nullptr if the builtin is not found.
  bool required) {

  const auto& variants = getAllOperatorsFor(name);
  std::stringstream failure_messages;
  for (const std::shared_ptr<Operator>& op : variants) {
    if (auto result = tryEmitBuiltin(
            op, failure_messages, loc, graph, name, inputs, attributes)) {
      return result;
    }
  }
  // none of the options worked
  if(!required) {
    return nullptr;
  }
  if(variants.size() == 0) {
    throw ErrorReport(loc) << "unknown builtin op";
  }
  throw ErrorReport(loc) << "arguments for call are not valid:\n"
                         << prefixLine(failure_messages.str(), "  ")
                         << "for call at";
}

static Value* ensureTensor(const SourceRange& range, Value* v) {
  if(!v->type()->isSubtypeOf(DynamicType::get())) {
    throw ErrorReport(range) << "expected a tensor value but found a "
                             << v->type()->str();
  }
  return v;
}

static Value* ensureInt(const SourceRange& range, Value* v) {
  if(!v->type()->isSubtypeOf(IntType::get())) {
    throw ErrorReport(range) << "expected a int but found a "
                             << v->type()->str();
  }
  return v;
}


void ensureTensors(const SourceRange& range, at::ArrayRef<Value*> values) {
  for(auto value : values) {
    ensureTensor(range, value);
  }
}

static Value* identity(const SourceRange& range, Value* v) {
  return v;
}


std::shared_ptr<SugaredValue> BuiltinFunction::call(
    SourceRange loc,
    Method & m,
    at::ArrayRef<NamedValue> inputs_,
    at::ArrayRef<NamedValue> attributes,
    size_t n_binders) {
  std::vector<NamedValue> inputs;
  if (value)
    inputs.push_back(*value);
  inputs.insert(inputs.end(), inputs_.begin(), inputs_.end());
  return std::make_shared<SimpleValue>(emitBuiltinCall(loc, *m.graph(), Symbol::aten(name), inputs, attributes, true));
}

struct to_ir {
  to_ir(
      Def def,
      FunctionTable& function_table,
      const Resolver& resolver,
      SugaredValuePtr self,
      Method& method) // method being constructed
      : method(method)
      , graph(method.graph())
      , def(def)
      , function_table(function_table)
      , resolver(resolver)
      , environment_stack(nullptr) {
    pushFrame(graph->block());

    auto schema = extractSchemaFromDef(def, bool(self));

    std::vector<Argument> arguments, returns; // for schema
    // inputs
    auto it = def.decl().params().begin();
    auto end = def.decl().params().end();
    // Type annotations exclude explicitly typing the "self" parameter, so in the
    // case that this is a method with self we expect one fewer parameter annotation
    // than the number of parameters this Def takes.
    if (self && def.decl().params().size() == 0) {
      throw ErrorReport(def.decl().params().range()) << "methods must have a self argument";
    }
    auto expected_annotation_size = self ? def.decl().params().size() - 1 : def.decl().params().size();
    if (schema.arguments.size() != expected_annotation_size) {
      throw ErrorReport(def.decl().params().range()) << "Number of type annotations for"
        << " function parameters (" << arguments.size() << ")"
        << " does not match the number of parameters on the function ("
        << expected_annotation_size << ")!";
    }
    if(self) {
      if(it == end)
        throw ErrorReport(def.decl().params().range()) << "methods must have a self argument";
      environment_stack->setSugaredVar(def.range(), (*it).ident().name(), self);
      ++it;
    }
    size_t arg_annotation_idx = 0;
    for(;it != end; ++it) {
      auto& name = (*it).ident().name();
      // Add the input to the graph
      Value *new_input = graph->addInput(name);
      environment_stack->setVar((*it).ident().range(), name, new_input);

      // Record the type for the schema and set the Type on the Value*
      arguments.push_back(schema.arguments.at(arg_annotation_idx++));
      new_input->setType(arguments.back().type);
    }
    // body
    auto stmts = def.statements();
    auto stmts_begin = stmts.begin();
    auto stmts_end = stmts.end();
    bool has_return = false;
    if (stmts_begin != stmts_end && (*std::prev(stmts_end)).kind() == TK_RETURN) {
      --stmts_end;
      has_return = true;
    }

    emitStatements(stmts_begin, stmts_end);

    // outputs
    if (has_return) {
      auto return_stmt = Return(*stmts_end);
      auto results = getValues(return_stmt.values(), true, identity);
      // a single return value that is a tuple expands in place:
      // return a
      if (return_stmt.values().size() == 1 && results.size() == 1) {
        auto result = results.at(0);
        if(result->type()->cast<TupleType>()) {
          results = createTupleUnpack(result).vec();
        }
      }
      if (!schema.is_varret && schema.returns.size() != results.size()) {
        throw ErrorReport(def.range()) << "Number of type annotations for function"
          << " return (" << schema.returns.size() << ") does not match"
          << " the number of returns from the function (" << results.size() << ")!";
      }
      auto range = return_stmt.range();
      size_t return_type_idx = 0;
      for (auto& r : results) {
        // TODO: support tuples and lists as returns
        auto return_kind = r->type()->kind();
        if (return_kind != TypeKind::TensorType &&
            return_kind != TypeKind::DynamicType &&
            return_kind != TypeKind::IntType &&
            return_kind != TypeKind::FloatType) {
          throw ErrorReport(return_stmt.range()) << "The only supported return types "
            << "are tensors, ints and floats";
        }
        graph->registerOutput(r);
        TypePtr type = DynamicType::get();
        if (!schema.is_varret) {
          type = schema.returns.at(return_type_idx).type;
          if (!r->type()->isSubtypeOf(type)) {
            throw ErrorReport(return_stmt.range()) << "Return value at position "
              << return_type_idx << " was annotated as having type " << type->str()
              << " but is actually of type " << r->type()->str();
          }
          return_type_idx++;
        }
        returns.push_back({"", type});
      }
    }

    method.setSchema({def.name().name(), std::move(arguments), std::move(returns)});
    // remove any uses of tuples that we inserted
    LowerTuples(graph);
  }

private:
  Method& method;
  std::shared_ptr<Graph> graph;
  Def def;
  FunctionTable& function_table;
  const Resolver& resolver;

  // Singly-linked list of environments. This top element contains a member
  // `next` that points to the most immediate enclosing scope's value.
  std::shared_ptr<Environment> environment_stack;

  void pushFrame(Block * b) {
    environment_stack = std::make_shared<Environment>(method, resolver, b, environment_stack);
  }
  std::shared_ptr<Environment> popFrame() {
    auto old_frame = environment_stack;
    environment_stack = environment_stack->next;
    return old_frame;
  }
  void emitStatements(const List<Stmt>& statements) {
    return emitStatements(statements.begin(), statements.end());
  }
  void emitStatements(List<Stmt>::const_iterator begin, List<Stmt>::const_iterator end) {
    for (; begin != end; ++begin) {
      auto stmt = *begin;
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
        case TK_GLOBAL:
          for (auto ident : Global(stmt).names()) {
            const auto& name = Ident(ident).name();
            environment_stack->setVar(ident.range(), name, graph->addInput(name));
          }
          break;
        case TK_EXPR_STMT: {
          auto exprs = ExprStmt(stmt).exprs();
          for (const auto& expr : exprs) {
            emitSugaredExpr(expr, 0);
          }
        }
        break;
        case TK_RETURN:
          throw ErrorReport(stmt) << "return statements can appear only at the end "
                                  << "of the function body";
          break;
      }
    }
  }

  std::shared_ptr<Environment> emitSingleIfBranch(
      Block* b,
      const List<Stmt> branch) {
    pushFrame(b);
    WithInsertPoint guard(b);
    emitStatements(branch);
    return popFrame();
  }

  Node* create(Symbol kind, const SourceRange& loc,  size_t n_outputs) {
    return graph
             ->create(kind, n_outputs)
             ->setSourceLocation(std::make_shared<SourceRange>(loc));
  }

  Value* emitTernaryIf(const TernaryIf& expr) {
    Value* cond_value = emitCond(expr.cond());

    Node* n = graph->insertNode(create(prim::If, expr.range(), 0));

    n->addInput(cond_value);
    auto* true_block = n->addBlock();
    auto* false_block = n->addBlock();


    auto emit_if_expr = [this](Block* b, const Expr& expr) {
      pushFrame(b);
      WithInsertPoint guard(b);
      Value* out_val = emitExpr(expr);
      b->registerOutput(out_val);
      popFrame();
    };

    emit_if_expr(true_block, expr.true_expr());
    emit_if_expr(false_block, expr.false_expr());

    auto true_type = unshapedType(true_block->outputs().at(0)->type());
    auto false_type = unshapedType(false_block->outputs().at(0)->type());
    if (*true_type != *false_type) {
      throw ErrorReport(expr)
          << "if-expression's true branch has type " << true_type->str()
          << " but false branch has type " << false_type->str();
    }

    // Add op outputs
    auto expr_value = n->addOutput()->setType(true_type); // Resulting value

    return expr_value;
  }

  Value* emitCond(Expr cond) {
    Value* v = emitExpr(cond, identity);
    if(v->type()->isSubtypeOf(DynamicType::get())) {
      v = typeCast(cond.range(), v, IntType::get());
    }
    if(!v->type()->isSubtypeOf(IntType::get())) {
      throw ErrorReport(cond) << "expected a tensor or integer expression for condition but found " << v->type()->str();
    }
    return v;
  }

  void emitIf(const If& stmt) {
    Value* cond_value = emitCond(stmt.cond());

    Node* n = graph->insertNode(create(prim::If, stmt.range(), 0));
    n->addInput(cond_value);
    auto* true_block = n->addBlock();
    auto* false_block = n->addBlock();

    // Emit both blocks once to get the union of all mutated values
    auto save_true = emitSingleIfBranch(true_block, stmt.trueBranch());
    auto save_false = emitSingleIfBranch(false_block, stmt.falseBranch());

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


    //ordered set, because we want deterministic graph output
    std::set<std::string> mutated_variables;

    for(auto & v : save_true->definedVariables()) {
      if(save_false->findInAnyFrame(v)) {
        mutated_variables.insert(v);
      }
    }
    for(auto & v : save_false->definedVariables()) {
      if(save_true->findInAnyFrame(v)) {
        mutated_variables.insert(v);
      }
    }

    // Register outputs in each block
    for (const auto& x : mutated_variables) {
      auto tv = save_true->getVar(x, stmt.range());
      true_block->registerOutput(tv);
      auto fv = save_false->getVar(x, stmt.range());
      false_block->registerOutput(fv);
      environment_stack->setVar(stmt.range(), x, n->addOutput()->setType(tv->type()));
    }

  }

  // *********************** Loop Operators ************************************
  // Emits a loop operators conforming to the semantics specified at
  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#experimental-loop
  // TODO: implement scan_outputs

  // the format of the Loop instruction is:
  // loop_carried_outputs* = Loop(max_trip_count, start_condition,
  // loop_carried_inputs*)
  //                          block0(loop_counter, loop_carried_block*) {
  //                             <body>
  //                             -> (continue_condition,
  //                             loop_carried_block_outputs*)
  //                          }
  // all loop_carried_... lists are the same length and represent the value of
  // loop-carried variables whose definitions are updated as the loop executes
  // in a way that ensure single static assignment.


  void emitLoopCommon(
      SourceRange range,
      at::optional<Expr> max_trip_count,
      at::optional<Expr> cond,
      const List<Stmt>& body,
      at::optional<Ident> itr_ident) {
    Node* n = graph->insertNode(create(prim::Loop, range, 0));
    Value *max_trip_count_val, *cond_val;
    {
      WithInsertPoint guard(n);
      if (max_trip_count) {
        max_trip_count_val = emitExpr(max_trip_count.value(), ensureInt);
      } else {
        max_trip_count_val =
            graph->insertConstant(INT_MAX,range);
      }
      if (cond) {
        cond_val = emitCond(cond.value());
      } else {
        cond_val = graph->insertConstant(true, range);
      }
    }
    n->addInput(max_trip_count_val);
    n->addInput(cond_val);
    auto* body_block = n->addBlock();
    Value* trip_count = body_block->addInput()->setType(IntType::get()); // Iteration num

    {
      pushFrame(body_block);
      if (itr_ident) {
        environment_stack->setVar(itr_ident->range(), itr_ident->name(), trip_count);
      }
      WithInsertPoint guard(body_block);
      emitStatements(body);

      // Also emit the conditional
      if (cond) {
        Value* body_cond_value = emitCond(cond.value());
        body_block->registerOutput(body_cond_value);
      } else {
        Value* cond_value_dummy = graph->insertConstant(true, range);
        body_block->registerOutput(cond_value_dummy);
      }

      auto body_frame = popFrame();
      auto outer_frame = environment_stack;

      // Add block outputs to correspond to each captured input
      // some of these will be removed.
      for (const auto& x : body_frame->captured_inputs) {
        auto fv = body_frame->getValueInThisFrame(range, x);
        body_block->registerOutput(fv);
      }

      // Remove inputs for values that did not mutate within the
      // block
      body_frame->deleteExtraInputs(range);

      // register node inputs/outputs for the true loop carried deps,
      for(size_t i = 0; i < body_frame->captured_inputs.size(); ++i) {
        auto x = body_frame->captured_inputs[i];
        n->addInput(outer_frame->getVar(x, range));
        // body_block->inputs(): loop_counter, lcd0, lcd1, ...
        // captured_inputs: lcd0, lcd1, ...
        auto typ = body_block->inputs()[i + 1]->type();
        outer_frame->setVar(range, x, n->addOutput()->setType(typ));
      }

    }
  }

  void emitForRange(SourceRange range, const Ident& target, const List<Expr>& args, const List<Stmt>& body) {
    // TODO: start, stop, step loop
    if (args.size() != 1) {
      throw ErrorReport(range)
          << "range() expects 1 argument but got " << args.size();
    }
    emitLoopCommon(range, {args[0]}, {}, body, target);
  }

  void emitFor(const For& stmt) {
    // For now, we only support range loops. e.g. for i in range(3): ...
    auto targets = stmt.targets();
    auto itrs = stmt.itrs();
    auto body = stmt.body();

    if (stmt.itrs().size() != 1) {
      throw ErrorReport(stmt)
          << "List of iterables is not supported currently.";
    }
    if (targets.size() != 1) {
      throw ErrorReport(stmt) << "Iteration variable unpacking is not supported";
    }

    if (targets[0].kind() != TK_VAR) {
      throw ErrorReport(targets[0]) << "Starred unpacking is currently not"
          << " supported for for loops.";
    }
    auto target = Var(targets[0]).name();

    // match range(<expr>) style loops
    // itrs must consist of a single Apply node
    if (itrs[0].kind() == TK_APPLY) {
      Apply range_iterator = Apply(itrs[0]);
      if (range_iterator.callee().kind() == TK_VAR) {
        Var var = Var(range_iterator.callee());
        if (var.name().name() == "range") {
          return emitForRange(stmt.range(), target, range_iterator.inputs(), body);
        }
      }
    }

    // it isn't a range(<expr>) loop, treat it as a sugared value that maybe can be
    // unrolled
    auto sv = emitSugaredExpr(itrs[0], 1);
    auto instances = sv->asTuple(stmt.range(), method);
    const std::string& target_name = target.name();
    pushFrame(environment_stack->block());
    for(auto inst : instances) {
      environment_stack->setSugaredVar(itrs[0].range(), target_name, inst);
      emitStatements(body);
    }

    for (const auto & n : environment_stack->definedVariables()) {
      if (environment_stack->findInParentFrame(n)) {
        environment_stack->next->setVar(stmt.range(), n, environment_stack->getVar(n, stmt.range()));
      }
    }
    popFrame();
  }

  void emitWhile(const While& stmt) {
    auto cond = stmt.cond();
    emitLoopCommon(stmt.range(), {}, {cond}, stmt.body(), {});
  }

  // Validate that the `lhs` Expr's in an assignment statement are valid. That
  // is:
  //
  // 1) All lhs Expr's are either Var or Starred nodes
  // 2) There is at most one Starred node in the lhs Expr
  // 3) A Starred node can only appear when there is another non-Starred lhs Expr
  //    Concretely this means that `*abc = func()` is illegal. Unpacking all
  //    outputs into a tuple is covered by `abc = func()`.
  bool calcNumStarredUnpack(const List<Expr>& lhs, const SourceRange& r) {
    size_t num_normal_assign = 0;
    size_t num_starred = 0;
    for (const auto& assignee : lhs) {
      if (assignee.kind() == TK_VAR) {
        num_normal_assign++;
      } else if (assignee.kind() == TK_STARRED) {
        num_starred++;
      } else {
        throw ErrorReport(assignee)
            << "lhs of assignment must be a variable or starred expression.";
      }
    }

    if (num_starred > 1) {
      throw ErrorReport(r)
          << "Only one starred expression is allowed on the lhs.";
    }

    if (num_starred > 0 && num_normal_assign == 0) {
      throw ErrorReport(r) << "A Starred expression may only appear on the "
                              << "lhs within the presence of another non-starred"
                              << " expression.";
    }

    return num_starred;
  }

  void emitAssignment(const Assign& stmt) {
    bool starred_unpack = calcNumStarredUnpack(stmt.lhs(), stmt.range());
    if (stmt.reduction() != '=') {
      if (stmt.lhs().size() != 1) {
        throw ErrorReport(stmt)
            << "reductions are only allowed when there is a single variable "
            << "on the left-hand side.";
      }
      Ident lhs = Var(stmt.lhs()[0]).name();
      Expr expr = BinOp::create(stmt.range(), stmt.reduction(),
                                Var::create(lhs.range(), lhs), stmt.rhs());
      environment_stack->setVar(lhs.range(), lhs.name(), emitExpr(expr));
      return;
    }

    // See [N_BINDERS]
    size_t n_binders = stmt.lhs().size();
    if(starred_unpack)
      n_binders--;

    auto output = emitSugaredExpr(stmt.rhs(), n_binders);

    if(stmt.lhs().size() == 1) {
      JIT_ASSERT(!starred_unpack);
      auto v = Var(stmt.lhs()[0]);
      environment_stack->setSugaredVar(v.range(), v.name().name(), output);
      return;
    }

    auto outputs = output->asTuple(stmt.rhs().range(), method);
    if(outputs.size() < n_binders) {
      throw ErrorReport(stmt)
        << "need " << (starred_unpack ? "at least " : "")
        << n_binders << " values to unpack but found only "
        << outputs.size();
    }
    if(outputs.size() > n_binders && !starred_unpack) {
      throw ErrorReport(stmt)
      << "too many values to unpack, need " << n_binders << " but found "
      << outputs.size();
    }
    int i = 0;
    for (auto assignee : stmt.lhs()) {
      if (assignee.kind() == TK_VAR) {
        environment_stack->setSugaredVar(assignee.range(), Var(assignee).name().name(), outputs.at(i));
        i++;
      } else if (assignee.kind() == TK_STARRED) {
        auto var = Starred(assignee).expr();
        if (var.kind() != TK_VAR) {
          throw ErrorReport(var) << "Cannot pack a tuple into a non-variable.";
        }
        size_t n_matched = outputs.size() - n_binders;
        ArrayRef<std::shared_ptr<SugaredValue>> outputs_ref = outputs;
        auto values = fmap(outputs_ref.slice(i, n_matched), [&](const std::shared_ptr<SugaredValue>& v) {
          return v->asValue(assignee.range(), method);
        });
        auto tup = graph->insertNode(graph->createTuple(values))->output();
        environment_stack->setVar(
          var.range(), Var(var).name().name(), tup);
        i += n_matched;
      }
    }
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
      case TK_NOT:
        return aten::__not__;
      default:
        throw std::runtime_error("unknown kind " + std::to_string(kind));
    }
  }



  std::vector<NamedValue> getNamedValues(
      TreeList trees,
      bool maybe_unpack=false,
      std::function<Value*(const SourceRange&, Value*)> post_process = ensureTensor) {
    std::vector<NamedValue> values;
    for (const auto& tree : trees) {
      if(maybe_unpack && tree->kind() == TK_STARRED) {
        auto starred = Starred(tree);
        auto entries = emitSugaredExpr(starred.expr(), 1)->asTuple(starred.range(), method);
        for(auto entry : entries) {
          values.push_back(NamedValue(
              tree->range(),
              post_process(
                  starred.range(), entry->asValue(starred.range(), method))));
        }
      } else {
        values.push_back(NamedValue(
            tree->range(), emitExpr(Expr(tree), post_process)));
      }
    }
    return values;
  }
  std::vector<NamedValue> getNamedValues(
      List<Expr> trees,
      bool maybe_unpack=false,
      std::function<Value*(const SourceRange&, Value*)> post_process = ensureTensor) {
    return getNamedValues(trees.tree()->trees(), maybe_unpack, post_process);
  }

  std::vector<Value*> getValues(
      TreeList trees,
      bool maybe_unpack=false,
      std::function<Value*(const SourceRange&, Value*)> post_process = ensureTensor) {
    return toValues(*graph, getNamedValues(trees, maybe_unpack, post_process));
  }
  std::vector<Value*> getValues(
      List<Expr> trees,
      bool maybe_unpack=false,
      std::function<Value*(const SourceRange&, Value*)> post_process = ensureTensor) {
    return getValues(trees.tree()->trees(), maybe_unpack, post_process);
  }

  // special rules apply when we directly call foo(a,b) when foo is an ident
  std::shared_ptr<SugaredValue> emitApplyIdent(Ident ident, const std::vector<NamedValue>& inputs, at::ArrayRef<NamedValue> attributes, size_t n_binders) {
    auto it = function_table.find(ident.name());
    if (it != function_table.end()) {
      return std::make_shared<SimpleValue>(packOutputs(*graph, method.emit_call_to(ident.range(), it->second, inputs, attributes)));
    }
    if(auto result = emitBuiltinCall(ident.range(), *method.graph(), Symbol::aten(ident.name()), inputs, attributes, false)) {
      return std::make_shared<SimpleValue>(result);
    }
    // it wasn't known built in, so treat it like standard apply
    return emitApplyExpr(Var::create(ident.range(), ident), inputs, attributes, n_binders);
  }

  std::shared_ptr<SugaredValue> emitApplyExpr(Expr callee, const std::vector<NamedValue>& inputs, at::ArrayRef<NamedValue> attributes, size_t n_binders) {
    // otherwise we evaluate the callee and then desugar it
    auto sv = emitSugaredExpr(callee, 1);
    return sv->call(callee.range(), method, inputs, attributes, n_binders);
  }

  Value* emitExpr(Expr tree, std::function<Value*(const SourceRange&, Value*)> post_process = ensureTensor) {
    return post_process(tree.range(), emitSugaredExpr(tree, 1)->asValue(tree.range(), method));
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
    throw std::runtime_error("reverseComparision: unsupported NodeKind. File a bug");
  }

  // any expression that can produce a SugaredValue is handled here
  // expressions that only return a single Value* are handled in emitSimpleExpr
  std::shared_ptr<SugaredValue> emitSugaredExpr(Expr tree, size_t n_binders) {
    switch(tree.kind()) {
      case TK_VAR:
        return environment_stack->getSugaredVar(Var(tree).name());
      case '.': {
        auto select = Select(tree);
        auto sv = emitSugaredExpr(select.value(), 1);
        return sv->attr(select.range(), method, select.selector().name());
      }
      case TK_APPLY: {
        auto apply = Apply(tree);
        auto inputs = getNamedValues(apply.inputs(), true, identity);
        auto attributes = fmap(apply.attributes(), [&](const Attribute& attr) {
          return NamedValue(attr.range(), attr.name().name(), emitExpr(attr.value(), identity));
        });
        // the apply is directly an identifier 'foo'
        if(apply.callee().kind() == TK_VAR) {
          return emitApplyIdent(Var(apply.callee()).name(), inputs, attributes, n_binders);
        }
        return emitApplyExpr(apply.callee(), inputs, attributes, n_binders);
      } break;
      default:
        return std::make_shared<SimpleValue>(emitSimpleExpr(tree));
    }
  }

  Value* emitSimpleExpr(
      const TreeRef& tree) {
    switch (tree->kind()) {
      case '@':
      case TK_POW:
      case TK_AND:
      case TK_OR:
      case TK_NOT:
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
      case TK_UNARY_MINUS: {
        const auto& inputs = tree->trees();
        auto kind = getNodeKind(tree->kind(), inputs.size());
        auto named_values = getNamedValues(inputs, /*maybe_unpack=*/false, identity);
        return emitBuiltinCall(
                   tree->range(),
                   *method.graph(),
                   kind,
                   named_values,
                   {},
                   /*required=*/true);
      }
      case TK_STARRED: {
        throw ErrorReport(tree) << "Unexpected starred expansion. File a bug report.";
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
        return emitNone(tree->range());
      } break;
      case TK_SUBSCRIPT: {
        const auto subscript = Subscript(tree);
        auto slice_exprs = subscript.subscript_exprs();
        if (slice_exprs.size() != 1) {
          // TODO Implement multidimensional slice
          throw ErrorReport(subscript.range()) << "Subscripting multiple dimensions is currently not supported\n";
        }
        if (slice_exprs[0].kind() == TK_SLICE_EXPR) {
          return emitSlice(subscript);
        } else {
          return emitGather(
              subscript.range(), TreeList{subscript.value(), slice_exprs[0]});
        }
      } break;
      case TK_IF_EXPR: {
        return emitTernaryIf(TernaryIf(tree));
      } break;
      case TK_STRINGLITERAL: {
        return emitStringLiteral(StringLiteral(tree));
      } break;
      case TK_LIST_LITERAL: {
        auto ll = ListLiteral(tree);
        auto values = getValues(ll.inputs(), /*maybe_unpack=*/true, identity);

        // If this is an empty list literal `[]`, construct an empty Tensor[]
        const auto elem_type =
            values.empty() ? DynamicType::get() : values.at(0)->type();
        for (auto v : values) {
          if (v->type() != elem_type) {
            throw ErrorReport(tree)
                << "Lists must contain only a single type, expected: "
                << *elem_type << " but found " << *v->type() << " instead";
          }
        }
        return graph->insertNode(graph->createList(elem_type, values))
            ->output();
      } break;
      case TK_TUPLE_LITERAL: {
        auto ll = TupleLiteral(tree);
        auto values = getValues(ll.inputs(), /*maybe_unpack=*/true, identity);
        return graph->insertNode(graph->createTuple(values))->output();
      } break;
      default:
        throw ErrorReport(tree) << "NYI: " << tree;
        break;
    }
  }

  Value* emitNone(SourceRange range) {
    auto& g = *method.graph();
    return g.insertNode(
        g.create(prim::None, {}, 1)->setSourceLocation(
          std::make_shared<SourceRange>(range)))->output();
  }

  Value* emitConst(const Const& c) {
    if (c.isFloatingPoint())
      return graph->insertConstant(c.asFloatingPoint(), c.range());
    else
      return graph->insertConstant(c.asIntegral(), c.range());
  }

  Value* emitStringLiteral(const StringLiteral& c) {
    return insertConstant(*graph, c.text(), c.range());
  }

  // Desugars slice syntactic sugar tensor[begin:end] -> tensor.slice(begin,
  // end).
  Value* emitSlice(const Subscript& subscript) {
    const auto& loc = subscript.range();
    // TODO: implement multidimensional slice
    JIT_ASSERT(subscript.subscript_exprs().size() == 1);
    JIT_ASSERT(subscript.subscript_exprs()[0].kind() == TK_SLICE_EXPR);
    auto slice_exp = SliceExpr(subscript.subscript_exprs()[0]);
    TreeList inputs = {subscript.value(), slice_exp.startOr(0)};
    const auto applyInputs = Compound::create(TK_LIST, loc, std::move(inputs));
    const auto input_values = getNamedValues(
        applyInputs->trees(),
        /*maybe_unpack*/ false,
        identity);

    NamedValue sliceable = input_values[0];
    NamedValue begin = input_values[1];
    NamedValue step = NamedValue(loc, "step", graph->insertConstant(1, loc));

    // Build the input arguments
    std::vector<NamedValue> args = {sliceable};
    if (sliceable.value(*graph)->type()->kind() == TypeKind::DynamicType) {
      // If the sliceable object is a tensor, specify a default dimension
      args.emplace_back(loc, "dim", graph->insertConstant(0, loc));
    }

    args.push_back(begin);

    const auto has_end = slice_exp.end().present();
    if (has_end) {
      // If the user specified an `end` index, pass it down
      args.emplace_back(loc, "end", emitExpr(Expr(slice_exp.end().get()), identity));
    }

    return emitBuiltinCall(loc, *graph, aten::slice, args, {step}, true);
}

  // Desugars gather syntactic sugar foo[i]
  Value* emitGather(
      const SourceRange& loc,
      TreeList&& inputs) {
    const auto applyInputs =
        Compound::create(TK_LIST, loc, std::move(inputs));
    auto input_values = getNamedValues(applyInputs->trees(),
                                        /*maybe_unpack*/false,
                                        identity);
    NamedValue gatherable = input_values[0];
    NamedValue idx = input_values[1];
    if (gatherable.value(*graph)->type()->kind() == TypeKind::ListType) {
      // if it's a list, emit a regular index selection op
      return emitBuiltinCall(
                 loc, *graph, aten::select, {gatherable, idx}, {}, true);

    } else {
      // if it's a single tensor, map tensor[idx] -> tensor.select(0, idx)
      NamedValue dim = NamedValue(loc, "dim", graph->insertConstant(0, loc));
      return emitBuiltinCall(
                 loc, *graph, aten::select, {gatherable, dim, idx}, {}, true);
    }
  }
};

static const std::unordered_map<std::string, std::string> &builtin_cast_methods() {
  static std::unordered_map<std::string, std::string> builtin_cast_methods = {
    {"byte", "_cast_Byte"},
    {"char", "_cast_Char"},
    {"double", "_cast_Double"},
    {"float", "_cast_Float"},
    {"int", "_cast_Int"},
    {"long", "_cast_Long"},
    {"short", "_cast_Short"},
    {"half", "_cast_Half"}
  };
  return builtin_cast_methods;
}

// support syntax sugar for x.foo(y, z) by allowing x.foo to return a
// callable value that will resolve to foo(x, y, z) when called.
std::shared_ptr<SugaredValue> SimpleValue::attr(SourceRange loc, Method & m, const std::string& field) {
  // Allow method-style casts on Tensor types. e.g. x.int()
  if (value->type()->isSubtypeOf(DynamicType::get()) && builtin_cast_methods().count(field)) {
    return std::make_shared<BuiltinFunction>(builtin_cast_methods().at(field), NamedValue(loc, "self", value));
  }
  return std::make_shared<BuiltinFunction>(field, NamedValue(loc, "self", value));
}

std::vector<Value*> inlineCallTo(Graph& g, Graph& callee, ArrayRef<Value*> inputs) {
  std::unordered_map<Value*, Value*> value_map;
  auto value_map_func = [&](Value* v) { return value_map.at(v); };
  JIT_ASSERT(callee.inputs().size() == inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    value_map[callee.inputs()[i]] = inputs[i];
  }
  for (auto* node : callee.nodes()) {
    auto* new_node =
        g.insertNode(g.createClone(node, value_map_func));
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      value_map[node->outputs()[i]] = new_node->outputs()[i];
    }
  }

  std::vector<Value*> outputs;
  for (auto* output : callee.outputs()) {
    outputs.push_back(value_map_func(output));
  }
  return outputs;
}

void defineMethodsInModule(Module & m, const std::vector<Def>& definitions, const std::vector<Resolver>& resolvers, SugaredValuePtr self) {
  FunctionTable table;
  JIT_ASSERT(definitions.size() == resolvers.size());
  auto resolver_it = resolvers.begin();
  std::vector<Method*> methods;
  for(Def def : definitions) {
    const std::string& name = def.name().name();
    Resolver resolver = *resolver_it++;
    auto creator = [def, &table, resolver, self](Method& method) {
      to_ir(def, table, resolver, self,  method);
    };
    Method& method = m.create_method(name, creator);
    // if self is defined, then these are methods and do not go into the global namespace
    // otherwise, they get defined together so we add them to the function table
    // so the methods can see each other
    if(!self) {
      auto result = table.emplace(name, method);
      JIT_ASSERT(result.second);
    }
    methods.push_back(&method);
  }
  for(Method* method : methods) {
    method->ensure_defined();
  }
}

const std::unordered_map<std::string, TypePtr> &ident_to_type_lut() {
  static std::unordered_map<std::string, TypePtr> map = {
    {"Tensor", DynamicType::get()},
    {"int", IntType::get()},
    {"float", FloatType::get()},
  };
  return map;
}

TypePtr parseTypeFromExpr(Expr expr);

const std::unordered_map<std::string, std::function<TypePtr(Subscript)>> &subscript_to_type_fns() {
  static std::unordered_map<std::string, std::function<TypePtr(Subscript)>> map = {
    {"Tuple", [](Subscript subscript) -> TypePtr {
      std::vector<TypePtr> subscript_expr_types;
      for (auto expr : subscript.subscript_exprs()) {
        subscript_expr_types.push_back(parseTypeFromExpr(expr));
      }
      return TupleType::create(subscript_expr_types);
    }},
  };
  return map;
}

TypePtr parseTypeFromExpr(Expr expr) {
  if (expr.kind() == TK_VAR) {
    auto ident = Var(expr).name();
    auto itr = ident_to_type_lut().find(ident.name());
    if (itr != ident_to_type_lut().end()) {
      return itr->second;
    }
    throw ErrorReport(expr.range()) << "Unknown type name " << ident.name();
  } else if (expr.kind() == TK_SUBSCRIPT) {
    auto subscript = Subscript(expr);
    if (subscript.value().kind() != TK_VAR) {
      throw ErrorReport(subscript.value().range()) << "Subscripted type must be a type identifier";
    }
    auto value_name = Var(subscript.value()).name().name();
    if (!subscript_to_type_fns().count(value_name)) {
      throw ErrorReport(subscript.range()) << "Type " << value_name << " cannot be subscripted";
    }
    return subscript_to_type_fns().at(value_name)(subscript);
  } else if (expr.kind() == '.') {
    auto select = Select(expr);
    if (select.value().kind() == TK_VAR && Var(select.value()).name().name() == "torch"
        && select.selector().name() == "Tensor") {
      return ident_to_type_lut().at("Tensor");
    }
  }
  throw ErrorReport(expr.range()) << "Expression of type " << kindToString(expr.kind())
                                  << " cannot be used in a type expression";
}

std::vector<Argument> parseArgsFromDecl(Decl decl, bool is_method) {
  std::vector<Argument> retval;
  size_t i = is_method ? 1 : 0;
  for (; i < decl.params().size(); ++i) {
    auto decl_arg = decl.params()[i];
    auto arg = Argument(decl_arg.ident().name(), parseTypeFromExpr(decl_arg.type()),
                        /*N =*/at::nullopt, /*default_value =*/at::nullopt,
                        /*kwarg_only =*/false);
    retval.push_back(arg);
  }
  return retval;
}

std::vector<Argument> parseReturnsFromDecl(Decl decl) {
  JIT_ASSERT(decl.return_type().present());
  auto parsed_type = parseTypeFromExpr(decl.return_type().get());
  if (auto tuple_type = parsed_type->cast<TupleType>()) {
    // Flatten a single return type of type Tuple into its constituent types
    std::vector<Argument> retval;
    for (auto type_ptr : tuple_type->elements()) {
      retval.emplace_back("", type_ptr, /*N =*/at::nullopt,
                          /*default_value =*/at::nullopt, /*kwarg_only =*/false);
    }
    return retval;
  } else {
    return {Argument("", parsed_type, /*N =*/at::nullopt,
                     /*default_value =*/at::nullopt, /*kwarg_only =*/false)};
  }
}

FunctionSchema extractSchemaFromDef(const Def &def, bool is_method) {
    auto name = def.name().name();
    std::vector<Argument> args = parseArgsFromDecl(def.decl(), is_method);
    std::vector<Argument> returns;
    bool is_varret;
    if (def.decl().return_type().present()) {
      returns = parseReturnsFromDecl(def.decl());
      is_varret = false;
    } else {
      is_varret = true;
    }
    return FunctionSchema(name, args, returns, false, is_varret);
}

void defineMethodsInModule(Module & m, const std::string& source, const Resolver& resolver, SugaredValuePtr self) {
  Parser p(source);
  std::vector<Def> definitions;
  std::vector<Resolver> resolvers;
  while (p.lexer().cur().kind != TK_EOF) {
    auto def = Def(p.parseFunction(/*is_method=*/bool(self)));
    definitions.push_back(def);
    resolvers.push_back(resolver);
  }
  defineMethodsInModule(m, definitions, resolvers, self);
}

std::shared_ptr<Graph> compileFunction(Def def, const Resolver& resolver) {
  Module m;
  defineMethodsInModule(m, {def}, {resolver}, nullptr);
  return m.get_method(def.name().name()).graph();
}

std::vector<std::shared_ptr<SugaredValue>> SimpleValue::asTuple(SourceRange loc, Method& m) {
  if(value->type()->kind() == TypeKind::TupleType) {
    auto outputs = createTupleUnpack(value);
    return fmap(outputs, [](Value* v) -> std::shared_ptr<SugaredValue> {
      return std::make_shared<SimpleValue>(v);
    });
  }
  throw ErrorReport(loc) << value->type()->str() << " cannot be used as a tuple";
}

void ensureSizeMatches(SourceRange loc, size_t expected, size_t actual, const std::string& what) {
  if(expected != actual) {
    throw ErrorReport(loc) << "expected " << expected << " " << what << " but found " << actual;
  }
}

} // namespace script
} // namespace jit
} // namespace torch
