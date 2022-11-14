#include <pybind11/detail/common.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/jit/api/object.h>
#include <torch/csrc/jit/python/script_init.h>
#include <torch/csrc/utils/pybind.h>

#include <caffe2/serialize/versions.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/jit/mobile/code.h>
#include <torch/csrc/jit/mobile/compatibility/backport.h>
#include <torch/csrc/jit/mobile/compatibility/model_compatibility.h>
#include <torch/csrc/jit/mobile/file_format.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/quantization.h>
#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <torch/csrc/jit/operator_upgraders/upgraders_entry.h>
#include <torch/csrc/jit/operator_upgraders/utils.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/python/python_ivalue.h>
#include <torch/csrc/jit/python/python_sugared_value.h>
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/testing/file_check.h>

#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_dict.h>
#include <torch/csrc/jit/python/python_list.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/logging.h>
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/testing/hooks_for_testing.h>

#include <torch/csrc/api/include/torch/ordered_dict.h>

#include <ATen/ATen.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/qualified_name.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/csrc/jit/mobile/train/export_data.h>
#include <chrono>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

using ::c10::Argument;
using ::c10::FunctionSchema;

using ResolutionCallback = std::function<py::object(std::string)>;
using FunctionDefaults = std::unordered_map<std::string, py::object>;
using ClassMethodDefaults = std::unordered_map<std::string, FunctionDefaults>;

namespace {

// A resolver that will inspect the outer Python scope to find `name`.
struct PythonResolver : public Resolver {
  explicit PythonResolver(ResolutionCallback rcb) : rcb_(std::move(rcb)) {}

  /**
   * While compiling classes, the class type we're compiling will not be
   * available in Python, since we haven't fowner_ defining the class yet. So
   * in order to make the class type available to its own methods, we need to
   * explicitly resolve it.
   *
   * @param rcb Python function to resolve a name to its Python object in the
   *            enclosing scope
   * @param classname The unqualified classname of the class currently being
   *                  compiled.
   * @param classType The class's type.
   */
  explicit PythonResolver(
      ResolutionCallback rcb,
      std::string classname,
      ClassTypePtr classType)
      : rcb_(std::move(rcb)),
        classname_(std::move(classname)),
        classType_(std::move(classType)) {}

  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      GraphFunction& m,
      const SourceRange& loc) override {
    pybind11::gil_scoped_acquire ag;
    py::object obj = rcb_(name);
    if (obj.is_none()) {
      return nullptr;
    }
    return toSugaredValue(obj, m, loc);
  }

  static bool isNamedTupleClass(py::object obj) {
    auto tuple_type = reinterpret_cast<PyObject*>(&PyTuple_Type);
    return PyObject_IsSubclass(obj.ptr(), tuple_type) &&
        py::hasattr(obj, "_fields");
  }

  TypePtr resolveTypeFromObject(const py::object& obj, const SourceRange& loc) {
    if (py::isinstance<ScriptClass>(obj)) {
      auto script_class = py::cast<ScriptClass>(obj);
      return script_class.class_type_.type_;
    }

    py::bool_ isClass = py::module::import("inspect").attr("isclass")(obj);
    if (!py::cast<bool>(isClass)) {
      return nullptr;
    }

    if (isNamedTupleClass(obj)) {
      return registerNamedTuple(obj, loc);
    }

    auto qualifiedName = c10::QualifiedName(
        py::cast<std::string>(py::module::import("torch._jit_internal")
                                  .attr("_qualified_name")(obj)));

    return get_python_cu()->get_type(qualifiedName);
  }

  TypePtr resolveType(const std::string& name, const SourceRange& loc)
      override {
    if (classType_ && name == classname_) {
      return classType_;
    }
    pybind11::gil_scoped_acquire ag;
    py::object obj = rcb_(name);
    if (obj.is_none()) {
      return nullptr;
    }

    auto annotation_type = py::module::import("torch.jit.annotations")
                               .attr("try_ann_to_type")(obj, loc);
    if (!annotation_type.is_none()) {
      return py::cast<TypePtr>(annotation_type);
    }
    return resolveTypeFromObject(obj, loc);
  }

 private:
  ResolutionCallback rcb_;
  std::string classname_;
  ClassTypePtr classType_;
};

std::shared_ptr<PythonResolver> pythonResolver(const ResolutionCallback& rcb) {
  return std::make_shared<PythonResolver>(rcb);
}
std::shared_ptr<PythonResolver> pythonResolver(
    const ResolutionCallback& rcb,
    std::string classname,
    ClassTypePtr classType) {
  return std::make_shared<PythonResolver>(
      rcb, std::move(classname), std::move(classType));
}

void checkOverloadDecl(const Decl& new_decl, const Decl& old_decl) {
  const auto& new_params = new_decl.params();
  const auto& old_params = old_decl.params();

  // TODO. same number of parameters not strictly necessary.
  TORCH_INTERNAL_ASSERT(
      new_params.size() == old_params.size(),
      "Overload must have same number of parameters\n",
      new_decl.range(),
      old_decl.range());
  for (const auto i : c10::irange(new_decl.params().size())) {
    TORCH_INTERNAL_ASSERT(
        new_params[i].ident().name() == old_params[i].ident().name(),
        "Overload parameters must have the same names\n",
        new_params[i].ident(),
        old_params[i].ident());
  }
}

c10::optional<IValue> tryCalculateDefaultParam(
    const Argument& arg,
    const py::object& def_value) {
  auto n = arg.N();
  auto list_type = arg.type()->cast<ListType>();
  try {
    if (n && *n > 0 && list_type) {
      // BroadcastingList, allow default values T for arg types List[T]
      return toIValue(def_value, list_type->getElementType());
    } else {
      return toIValue(def_value, arg.type());
    }
  } catch (...) {
    return c10::nullopt;
  }
}

// An overloaded function may have a default that does not subtype all overloads
// @overload
// def foo(x: str)
// def foo(x=1)
FunctionDefaults calcOverloadedFunctionDefaults(
    const FunctionSchema& schema,
    const FunctionDefaults& defaults) {
  FunctionDefaults updated_defaults;
  for (const auto& arg : schema.arguments()) {
    const std::string& arg_name = arg.name();
    auto value = defaults.find(arg_name);
    if (value == defaults.end()) {
      continue;
    }
    auto maybe_ivalue = tryCalculateDefaultParam(arg, value->second);
    if (maybe_ivalue) {
      updated_defaults[arg_name] = value->second;
    }
  }
  return updated_defaults;
}

} // namespace

bool checkMutableFunctionDefault(const py::object& def_arg) {
  if (py::isinstance<py::list>(def_arg) || py::isinstance<py::dict>(def_arg)) {
    return true;
  }
  if (py::isinstance<py::tuple>(def_arg)) {
    auto pytuple = def_arg.cast<py::tuple>();
    for (py::handle t : pytuple) {
      py::object obj = py::reinterpret_borrow<py::object>(t);
      if (checkMutableFunctionDefault(obj)) {
        return true;
      }
    }
  }
  return false;
}

void checkMutableFunctionDefault(
    const SourceRange& range,
    const Argument& arg,
    const py::object& def_arg) {
  if (checkMutableFunctionDefault(def_arg) || arg.type()->cast<ClassType>()) {
    throw ErrorReport(range)
        << "Mutable default parameters are not supported because Python binds them to the function"
        << " and they persist across function calls.\n As a workaround, make the default None and instantiate"
        << " the default parameter within the body of the function. Found "
        << def_arg.get_type() << " on parameter " << arg.name();
  }
}

FunctionSchema getSchemaWithNameAndDefaults(
    const SourceRange& range,
    const FunctionSchema& schema,
    const at::optional<std::string>& new_name,
    const FunctionDefaults& default_args) {
  std::vector<Argument> new_args;
  for (auto& arg : schema.arguments()) {
    auto it = default_args.find(arg.name());
    if (it != default_args.end()) {
      checkMutableFunctionDefault(range, arg, it->second);
      c10::optional<IValue> value = tryCalculateDefaultParam(arg, it->second);
      if (!value) {
        ErrorReport error(range);
        error << "Expected a default value of type " << arg.type()->repr_str()
              << " on parameter \"" << arg.name() << "\".";
        if (arg.is_inferred_type()) {
          error << "Because \"" << arg.name()
                << "\" was not annotated with an explicit type "
                << "it is assumed to be type 'Tensor'.";
        }
        throw error;
      }
      new_args.emplace_back(
          arg.name(), arg.type(), arg.N(), *value, arg.kwarg_only());
    } else {
      new_args.push_back(arg);
    }
  }
  return FunctionSchema(
      new_name.value_or(schema.name()),
      schema.overload_name(),
      new_args,
      schema.returns(),
      schema.is_vararg(),
      schema.is_varret());
}

static Decl mergeDefaultsAndExtraParametersToOverloadDecl(
    const Decl& overload_decl,
    const Decl& impl_decl,
    const FunctionDefaults& defaults) {
  std::vector<Param> adjusted_params;
  const auto& overload_params = overload_decl.params();
  const auto& impl_params = impl_decl.params();

  // following PEP specification that the following should work:
  // @overload
  // def mouse_event(x1: int, y1: int) -> ClickEvent: ...
  // ...
  // def mouse_event(x1: int, y1: int, x2: Optional[int] = None, y2:
  // Optional[int] = None)
  TORCH_CHECK(
      overload_params.size() <= impl_params.size(),
      "Overload should not have more parameters than implementation function",
      overload_decl.range(),
      impl_decl.range());

  for (const auto i : c10::irange(overload_params.size())) {
    auto overload_name = overload_params[i].ident().name();
    auto impl_name = impl_params[i].ident().name();
    if (overload_name != impl_name) {
      throw ErrorReport(overload_decl.range())
          << "Overload parameters must have the same names. "
          << "Found " << overload_name << " and " << impl_name
          << " on argument " << i;
    }
    adjusted_params.push_back(overload_params[i]);
  }
  for (size_t i = overload_params.size(); i < impl_params.size(); ++i) {
    if (!defaults.count(impl_params[i].ident().name())) {
      throw ErrorReport(impl_decl.range())
          << "Expected to find default parameter on argument"
          << impl_params[i].ident().name()
          << " because it is not defined on the overloaded declaration";
    }
    if (!impl_params[i].type().present()) {
      throw ErrorReport(impl_decl.range())
          << "Parameters not specified on the overloaded declaration must have a type annotation in the implementation function."
          << " Did not find type for param " << impl_params[i].ident().name();
    }
    adjusted_params.push_back(impl_params[i]);
  }
  return Decl::create(
      overload_decl.range(),
      List<Param>::create(overload_decl.range(), adjusted_params),
      overload_decl.return_type());
}

static StrongFunctionPtr script_compile_overloaded_function(
    const c10::QualifiedName& name,
    const Decl& overload_decl,
    const Def& implementation_def,
    const ResolutionCallback& rcb,
    const FunctionDefaults& implementation_defaults,
    const py::object& signature) {
  if (signature.is_none()) {
    throw ErrorReport(overload_decl.range())
        << "Must explicitly add type annotations to overloaded functions";
  }

  auto adjusted_decl = mergeDefaultsAndExtraParametersToOverloadDecl(
      overload_decl, implementation_def.decl(), implementation_defaults);
  auto new_def = implementation_def.withDecl(adjusted_decl);
  auto cu = get_python_cu();
  auto defined_functions = cu->define(
      QualifiedName(name.prefix()),
      /*properties=*/{},
      /*propResolvers=*/{},
      {new_def},
      {pythonResolver(rcb)},
      nullptr,
      true);
  TORCH_INTERNAL_ASSERT(defined_functions.size() == 1);
  auto& defined = defined_functions[0];
  FunctionDefaults updated_defaults = calcOverloadedFunctionDefaults(
      defined->getSchema(), implementation_defaults);
  defined->setSchema(getSchemaWithNameAndDefaults(
      new_def.range(),
      defined->getSchema(),
      new_def.name().name(),
      updated_defaults));
  StrongFunctionPtr ret(std::move(cu), defined);
  didFinishEmitFunction(ret);
  return ret;
}

static StrongFunctionPtr script_compile_function(
    const c10::QualifiedName& name,
    const Def& def,
    const FunctionDefaults& defaults,
    const ResolutionCallback& rcb) {
  auto cu = get_python_cu();
  auto defined_functions = cu->define(
      QualifiedName(name.prefix()),
      /*properties=*/{},
      /*propResolvers=*/{},
      {def},
      {pythonResolver(rcb)},
      nullptr,
      true);
  TORCH_INTERNAL_ASSERT(defined_functions.size() == 1);
  auto& defined = defined_functions[0];
  defined->setSchema(getSchemaWithNameAndDefaults(
      def.range(), defined->getSchema(), def.name().name(), defaults));
  StrongFunctionPtr ret(std::move(cu), defined);
  didFinishEmitFunction(ret);
  return ret;
}

struct VISIBILITY_HIDDEN ModuleSelf : public Self {
  ModuleSelf(std::shared_ptr<ConcreteModuleType> concreteType)
      : Self(), concreteType_(std::move(concreteType)) {}

  std::shared_ptr<SugaredValue> makeSugared(Value* v) const override {
    v->setType(getClassType());
    return std::make_shared<ModuleValue>(v, concreteType_);
  }

  ClassTypePtr getClassType() const override {
    return concreteType_->getJitType()->expect<ClassType>();
  }

 private:
  std::shared_ptr<ConcreteModuleType> concreteType_;
};

static TypePtr getTensorType(const at::Tensor& t, bool complete) {
  auto r = TensorType::create(t);
  if (!complete) {
    r = r->dimensionedOnly();
  }
  return r;
}

static TypePtr inferShapeAndTypeForInput(
    TypePtr input_type,
    Stack::const_iterator& s_iter,
    const Stack::const_iterator& s_iter_end,
    bool complete) {
  if (auto tuple_type = input_type->cast<TupleType>()) {
    std::vector<TypePtr> types;
    for (const auto& sub_type : tuple_type->containedTypes()) {
      TORCH_INTERNAL_ASSERT(s_iter != s_iter_end);
      types.emplace_back(
          inferShapeAndTypeForInput(sub_type, s_iter, s_iter_end, complete));
    }
    return TupleType::create(types);
  } else if (auto list_type = input_type->cast<ListType>()) {
    const TypePtr& sub_type = list_type->getElementType();
    auto elem_type =
        inferShapeAndTypeForInput(sub_type, s_iter, s_iter_end, complete);
    return ListType::create(elem_type);
  } else if (auto tensor_type = input_type->cast<TensorType>()) {
    auto type = getTensorType(s_iter->toTensor(), complete);
    s_iter++;
    return type;
  } else if (auto optional_type = input_type->cast<OptionalType>()) {
    const TypePtr& sub_type = optional_type->getElementType();
    auto elem_type =
        inferShapeAndTypeForInput(sub_type, s_iter, s_iter_end, complete);
    return OptionalType::create(elem_type);
  } else {
    // Primitive type, keep as is.
    s_iter++;
    return input_type;
  }
}

static void setInputTensorTypes(
    Graph& g,
    const Stack& stack,
    bool complete,
    const std::vector<int>& param_count_list = {}) {
  at::ArrayRef<Value*> input_values = g.inputs();
  auto s_iter = stack.begin();
  size_t list_idx = 0;
  if (!param_count_list.empty()) {
    TORCH_INTERNAL_ASSERT(
        input_values.size() == param_count_list.size(),
        " input_values:",
        input_values.size(),
        " vs param_count_list:",
        param_count_list.size());
  }
  for (auto v : input_values) {
    // Leave packed param types alone. This is needed for downstream passes
    // (like alias analysis) to work properly. This will be unpacked later
    // in unpackQuantizedWeights.
    if (auto named_type = v->type()->cast<c10::NamedType>()) {
      if (auto qualname = named_type->name()) {
        if (getCustomClass(qualname->qualifiedName())) {
          if (param_count_list.empty()) {
            AT_ASSERT(s_iter != stack.end());
            s_iter++;
          } else {
            if (param_count_list[list_idx] > 0) {
              AT_ASSERT(s_iter != stack.end());
            }
            s_iter += param_count_list[list_idx];
          }
          list_idx++;
          continue;
        }
      }
    }
    v->setType(
        inferShapeAndTypeForInput(v->type(), s_iter, stack.end(), complete));
    list_idx++;
  }
}

static std::shared_ptr<Graph> _propagate_shapes(
    Graph& graph,
    std::vector<at::Tensor> inputs,
    bool with_grad = false) {
  Stack stack(inputs.begin(), inputs.end());
  auto retval = graph.copy();
  setInputTensorTypes(*retval, stack, /*complete=*/false);
  PropagateInputShapes(retval);
  return retval;
}

static std::shared_ptr<Graph> _propagate_and_assign_input_shapes(
    Graph& graph,
    const std::vector<at::Tensor>& inputs,
    const std::vector<int>& param_count_list,
    bool with_grad = false,
    bool propagate = true) {
  auto retval = graph.copy();
  setInputTensorTypes(
      *retval, fmap<IValue>(inputs), /*complete=*/true, param_count_list);
  if (propagate) {
    PropagateInputShapes(retval);
  }
  return retval;
}

void addFunctionToModule(Module& module, const StrongFunctionPtr& func) {
  // Make a graph with a fake self argument
  auto graph = toGraphFunction(*func.function_).graph()->copy();
  auto v = graph->insertInput(0, "self");
  v->setType(module._ivalue()->type());
  const auto name = QualifiedName(*module.type()->name(), "forward");
  auto method =
      module._ivalue()->compilation_unit()->create_function(name, graph);
  module.type()->addMethod(method);
}

// this is used in our test suite to check that we correctly preserved type tags
bool ivalue_tags_match(const Module& lhs, const Module& rhs) {
  struct Work {
    IValue a;
    IValue b;
  };
  std::unordered_set<const void*> visited;
  std::vector<Work> work = {{lhs._ivalue(), rhs._ivalue()}};
  while (!work.empty()) {
    Work item = work.back();
    work.pop_back();
    if (item.a.isPtrType()) {
      // uncomment to debug type matching errors
      // std::cout << "MATCHING " << /*item.a <<*/ "(" << *item.a.type() << ") "
      //          << item.a.internalToPointer() << " " << /*item.b <<*/ " ("
      //          << *item.b.type() << ") " << item.b.internalToPointer() <<
      //          "\n";

      if (visited.count(item.a.internalToPointer())) {
        continue;
      }
      visited.emplace(item.a.internalToPointer());
    }
    if (!unshapedType(item.b.type())
             ->isSubtypeOf(unshapedType(item.b.type()))) {
      // Since named types are saved and loaded in the test suite, we cannot
      // expect them to be equal. We should still check their slots however.
      if (!item.a.type()->cast<c10::NamedType>()) {
        return false;
      }
    }
    // check tags for objects that contain subobjects
    if (item.a.isObject()) {
      auto ao = item.a.toObject();
      auto bo = item.b.toObject();
      for (size_t i = 0; i < ao->slots().size(); ++i) {
        work.emplace_back(Work{ao->slots().at(i), bo->slots().at(i)});
      }
    } else if (item.a.isTuple()) {
      auto at = item.a.toTuple();
      auto bt = item.b.toTuple();
      for (size_t i = 0; i < at->elements().size(); ++i) {
        work.emplace_back(Work{at->elements().at(i), bt->elements().at(i)});
      }
    } else if (item.a.isList()) {
      auto al = item.a.toList();
      auto bl = item.b.toList();
      for (const auto i : c10::irange(al.size())) {
        work.emplace_back(Work{al.get(i), bl.get(i)});
      }
    } else if (item.a.isGenericDict()) {
      auto ad = item.a.toGenericDict();
      auto bd = item.b.toGenericDict();
      for (auto& item : ad) {
        // Dictionaory keys cannot contain List/Dicts that require tags
        // so we do not have to check them.
        // Furthermore without ordered dicts it is expensive to find the
        // equivalent key
        work.emplace_back(Work{item.value(), bd.at(item.key())});
      }
    } else if (item.a.isFuture()) {
      auto af = item.a.toFuture();
      auto bf = item.b.toFuture();
      af->wait();
      bf->wait();
      work.emplace_back(Work{af->value(), bf->value()});
    }
  }

  return true;
}

// helper used to implement ._parameters, ._buffers, ._modules dicts
// inside of script nn.Module
template <typename Policy>
struct slot_dict_impl {
  slot_dict_impl(ModulePtr module) : module_(std::move(module)) {}
  bool contains(const std::string& name) const {
    if (auto slot = module_->type()->findAttributeSlot(name)) {
      if (Policy::valid(module_->type(), *slot, module_->getSlot(*slot))) {
        return true;
      }
    }
    return false;
  }

  std::vector<std::pair<std::string, py::object>> items() const {
    std::vector<std::pair<std::string, py::object>> result;
    for (size_t i = 0, N = module_->type()->numAttributes(); i < N; ++i) {
      if (Policy::valid(module_->type(), i, module_->getSlot(i))) {
        result.emplace_back(
            module_->type()->getAttributeName(i),
            toPyObject(module_->getSlot(i)));
      }
    }
    return result;
  }

  void setattr(const std::string& name, py::object value) {
    const TypePtr& type = module_->type()->getAttribute(name);
    Module(module_).setattr(name, toIValue(std::move(value), type));
  }

  py::object getattr(const std::string& name) {
    return toPyObject(Module(module_).attr(name));
  }

  static void bind(const py::module& m, const char* name) {
    py::class_<slot_dict_impl<Policy>>(m, name)
        .def(py::init(
            [](Module& m) { return slot_dict_impl<Policy>(m._ivalue()); }))
        .def("contains", &slot_dict_impl<Policy>::contains)
        .def("items", &slot_dict_impl<Policy>::items)
        .def("setattr", &slot_dict_impl<Policy>::setattr)
        .def("getattr", &slot_dict_impl<Policy>::getattr);
  }

 private:
  ModulePtr module_;
};

template <typename T>
py::list debugMakeList(const T& list) {
  py::list result;
  for (const auto& elem : list) {
    result.append(py::cast(elem));
  }
  return result;
}
template <typename T>
py::list debugMakeNamedList(const T& list) {
  py::list result;
  for (auto elem : list) {
    result.append(py::cast(std::make_pair(elem.name, elem.value)));
  }
  return result;
}
template <typename T>
py::set debugMakeSet(const T& list) {
  py::set result;
  for (const auto& elem : list) {
    result.add(py::cast(elem));
  }
  return result;
}

static py::dict _jit_debug_module_iterators(Module& module) {
  py::dict result;
  result["children"] = debugMakeList(module.children());
  result["named_children"] = debugMakeNamedList(module.named_children());
  result["modules"] = debugMakeList(module.modules());
  result["named_modules"] = debugMakeNamedList(module.named_modules());

  result["parameters"] = debugMakeList(module.parameters(false));
  result["named_parameters"] =
      debugMakeNamedList(module.named_parameters(false));
  result["parameters_r"] = debugMakeList(module.parameters(true));
  result["named_parameters_r"] =
      debugMakeNamedList(module.named_parameters(true));

  result["buffers"] = debugMakeList(module.buffers(false));
  result["named_buffers"] = debugMakeNamedList(module.named_buffers(false));
  result["buffers_r"] = debugMakeList(module.buffers(true));
  result["named_buffers_r"] = debugMakeNamedList(module.named_buffers(true));

  result["named_attributes"] =
      debugMakeNamedList(module.named_attributes(false));
  result["named_attributes_r"] =
      debugMakeNamedList(module.named_attributes(true));
  return result;
}

static constexpr std::array<const char*, 47> magic_method_names = {
    "__lt__",      "__le__",      "__eq__",        "__ne__",
    "__ge__",      "__gt__",      "__not__",       "__abs__",
    "__add__",     "__and__",     "__floordiv__",  "__index__",
    "__inv__",     "__invert__",  "__lshift__",    "__mod__",
    "__mul__",     "__matmul__",  "__neg__",       "__or__",
    "__pos__",     "__pow__",     "__rshift__",    "__sub__",
    "__truediv__", "__xor__",     "__concat__",    "__contains__",
    "__delitem__", "__getitem__", "__setitem__",   "__iadd__",
    "__iand__",    "__iconcat__", "__ifloordiv__", "__ilshift__",
    "__imod__",    "__imul__",    "__imatmul__",   "__ior__",
    "__ipow__",    "__irshift__", "__isub__",      "__itruediv__",
    "__ixor__",    "__str__",     "__len__",
};

struct DeepCopyMemoTable {
  std::shared_ptr<IValue::HashAliasedIValueMap> map;
};

IValue pyIValueDeepcopy(const IValue& ivalue, const py::dict& memo) {
  if (!memo.contains(py::str("__torch_script_memo_table"))) {
    memo["__torch_script_memo_table"] =
        DeepCopyMemoTable{std::make_shared<IValue::HashAliasedIValueMap>()};
  }
  auto& ivalue_memo =
      *py::cast<DeepCopyMemoTable>(memo["__torch_script_memo_table"]).map;
  return ivalue.deepcopy(ivalue_memo);
}

ExtraFilesMap extra_files_from_python(const py::dict& pydict) {
  ExtraFilesMap r;
  for (const auto& it : pydict) {
    r[py::cast<std::string>(it.first)] = "";
  }
  return r;
}

void extra_files_to_python(const ExtraFilesMap& m, const py::dict& pydict) {
  // py::dict is pointer-like type so it gets modified despite const&
  for (const auto& it : m) {
    pydict[py::str(it.first)] = py::bytes(it.second);
  }
}

void pyCompilationUnitDefine(
    CompilationUnit& cu,
    const std::string& src,
    const ResolutionCallback* rcb,
    const uint32_t _frames_up) {
  if (rcb && *rcb) {
    cu.define(c10::nullopt, src, pythonResolver(*rcb), nullptr);
  } else {
    py::object py_default_rcb =
        py::module::import("torch._jit_internal")
            .attr("createResolutionCallbackFromFrame")(_frames_up);
    auto default_rcb = py_default_rcb.cast<ResolutionCallback>();
    cu.define(c10::nullopt, src, pythonResolver(default_rcb), nullptr);
  }
}

void initJitScriptBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<c10::Capsule>(m, "Capsule");

  auto object_class =
      py::class_<Object>(m, "ScriptObject")
          .def("_type", [](Module& m) { return m.type(); })
          .def(
              "_get_method",
              [](Object& self, const std::string& name) -> Method {
                return self.get_method(name);
              },
              py::keep_alive<0, 1>())
          .def(
              "setattr",
              [](Object& self, const std::string& name, py::object value) {
                if (self.type()->hasConstant(name)) {
                  TORCH_CHECK(
                      false,
                      "Can't set constant '",
                      name,
                      "' which has value:",
                      self.type()->getConstant(name));
                }
                TypePtr type = self.type()->getAttribute(name);
                try {
                  auto ivalue = toIValue(std::move(value), type);
                  self.setattr(name, ivalue);
                } catch (std::exception& e) {
                  throw py::cast_error(c10::str(
                      "Could not cast attribute '",
                      name,
                      "' to type ",
                      type->repr_str(),
                      ": ",
                      e.what()));
                }
              })
          .def(
              "getattr",
              [](Object& self, const std::string& name) {
                try {
                  return toPyObject(self.attr(name));
                } catch (const ObjectAttributeError& err) {
                  throw AttributeError("%s", err.what());
                }
              })
          .def(
              "__getattr__",
              [](Object& self, const std::string& name) -> py::object {
                try {
                  if (name == "__qualname__") {
                    return py::cast(self.type()->name()->name());
                  }
                  if (auto method = self.find_method(name)) {
                    return py::cast(*method);
                  }
                  if (self.has_property(name)) {
                    auto prop = self.get_property(name);
                    // wrap the Method into callable PyObject
                    auto getter_func = py::cast(prop.getter_func);
                    return getter_func();
                  }
                  return toPyObject(self.attr(name));
                } catch (const ObjectAttributeError& err) {
                  throw AttributeError("%s", err.what());
                }
              })
          .def(
              "__setattr__",
              [](Object& self, const std::string& name, py::object value) {
                try {
                  if (self.has_property(name)) {
                    auto prop = self.get_property(name);
                    if (!prop.setter_func.has_value()) {
                      TORCH_CHECK(false, "can't set attribute");
                    }
                    // wrap the Method into callable PyObject
                    auto setter_func = py::cast(prop.setter_func);
                    setter_func(value);
                    return;
                  }

                  if (self.type()->hasConstant(name)) {
                    TORCH_CHECK(
                        false,
                        "Can't set constant '",
                        name,
                        "' which has value:",
                        self.type()->getConstant(name));
                  }
                  TypePtr type = self.type()->getAttribute(name);
                  auto ivalue = toIValue(std::move(value), type);
                  self.setattr(name, ivalue);
                } catch (const ObjectAttributeError& err) {
                  throw AttributeError("%s", err.what());
                }
              })
          .def(
              "hasattr",
              [](Object& self, const std::string& name) {
                return self.hasattr(name);
              })
          .def(
              "_has_method",
              [](Object& self, const std::string& name) {
                return bool(self.find_method(name));
              })
          .def(
              "_method_names",
              [](Object& self) {
                return fmap(self.get_methods(), [](const Method& method) {
                  return method.name();
                });
              })
          .def(
              "_properties", [](Object& self) { return self.get_properties(); })
          .def("__copy__", &Object::copy)
          .def(
              "__hash__",
              [](const Object& self) {
                // Similar to Tensor's `__hash__`, which is `id()`.
                return std::hash<c10::ivalue::Object*>{}(self._ivalue().get());
              })
          .def(py::pickle(
              [](const Object& self)
                  -> std::tuple<py::object, std::string> { // __getstate__
                if (auto getstate_method = self.find_method("__getstate__")) {
                  auto object_state = toPyObject((*getstate_method)(Stack{}));
                  TORCH_INTERNAL_ASSERT(self.type()->name());
                  return std::make_tuple(
                      object_state, self.type()->name()->qualifiedName());
                }
                std::stringstream err;
                err << "Tried to serialize object ";
                if (auto qualname = self.type()->name()) {
                  err << qualname->qualifiedName() << " ";
                }
                err << "which does not have a __getstate__ method defined!";
                throw std::runtime_error(err.str());
              },
              [](const std::tuple<py::object, std::string>& state_tup)
                  -> Object {
                py::object state;
                std::string qualname;
                std::tie(state, qualname) = state_tup;
                auto class_type = getCustomClass(qualname);
                TORCH_CHECK(
                    class_type,
                    "Tried to deserialize class ",
                    qualname,
                    " which is not known to the runtime. "
                    "If this is a custom C++ class, make "
                    "sure the appropriate code is linked.");

                auto self = Object(c10::ivalue::Object::create(
                    c10::StrongTypePtr(
                        std::shared_ptr<torch::jit::CompilationUnit>(),
                        class_type),
                    1));
                if (auto setstate_method = self.find_method("__setstate__")) {
                  auto setstate_schema =
                      setstate_method->function().getSchema();
                  TORCH_INTERNAL_ASSERT(
                      setstate_schema.arguments().size() == 2,
                      "__setstate__ method for class ",
                      class_type->repr_str(),
                      " must have exactly 2 arguments!");
                  auto state_type = setstate_schema.arguments().at(1).type();
                  (*setstate_method)(Stack{toIValue(state, state_type)});
                  return self;
                }
                std::stringstream err;
                err << "Tried to deserialize object ";
                if (auto qualname = class_type->name()) {
                  err << qualname->qualifiedName() << " ";
                }
                err << "which does not have a __setstate__ method defined!";
                throw std::runtime_error(err.str());
              }));

  py::class_<Object::Property>(m, "ScriptObjectProperty")
      .def_property_readonly(
          "name", [](const Object::Property& self) { return self.name; })
      .def_property_readonly(
          "getter",
          [](const Object::Property& self) { return self.getter_func; })
      .def_property_readonly("setter", [](const Object::Property& self) {
        return self.setter_func;
      });

  // Special case __str__ to make sure we can print Objects/Modules
  // regardless of if the user defined a __str__
  using MagicMethodImplType = std::function<py::object(
      const Object& self, py::args args, py::kwargs kwargs)>;
  std::unordered_map<std::string, MagicMethodImplType> special_magic_methods{
      {"__str__",
       [](const Object& self, py::args args, py::kwargs kwargs) -> py::object {
         auto method = self.find_method("__str__");
         if (!method) {
           return py::str("ScriptObject");
         }
         return invokeScriptMethodFromPython(
             *method,
             // NOLINTNEXTLINE(performance-move-const-arg)
             std::move(args),
             // NOLINTNEXTLINE(performance-move-const-arg)
             std::move(kwargs));
       }}};

  for (const char* mm_name : magic_method_names) {
    if (special_magic_methods.count(mm_name)) {
      object_class.def(mm_name, special_magic_methods[mm_name]);
    } else {
      object_class.def(
          mm_name,
          [mm_name](const Object& self, py::args args, py::kwargs kwargs) {
            auto method = self.find_method(mm_name);
            if (!method) {
              throw NotImplementedError();
            }
            return invokeScriptMethodFromPython(
                *method,
                // NOLINTNEXTLINE(performance-move-const-arg)
                std::move(args),
                // NOLINTNEXTLINE(performance-move-const-arg)
                std::move(kwargs));
          });
    }
  }

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<DeepCopyMemoTable>(m, "DeepCopyMemoTable");

  py::class_<UpgraderEntry>(m, "_UpgraderEntry")
      .def(py::init<int, std::string, std::string>())
      .def_property_readonly(
          "bumped_at_version",
          [](const UpgraderEntry& self) { return self.bumped_at_version; })
      .def_property_readonly(
          "upgrader_name",
          [](const UpgraderEntry& self) { return self.upgrader_name; })
      .def_property_readonly("old_schema", [](const UpgraderEntry& self) {
        return self.old_schema;
      });

  py::class_<UpgraderRange>(m, "_UpgraderRange")
      .def(py::init<int, int>())
      .def_property_readonly(
          "min_version",
          [](const UpgraderRange& self) { return self.min_version; })
      .def_property_readonly("max_version", [](const UpgraderRange& self) {
        return self.max_version;
      });

  object_class.def(
      "__deepcopy__", [](const Object& self, const py::dict& memo) {
        return Object(
            pyIValueDeepcopy(IValue(self._ivalue()), memo).toObject());
      });

  // Used by torch.package to save ScriptModule objects in unified format.
  py::class_<ScriptModuleSerializer>(m, "ScriptModuleSerializer")
      .def(py::init<caffe2::serialize::PyTorchStreamWriter&>())
      .def("serialize", &ScriptModuleSerializer::serialize_unified_format)
      .def(
          "write_files",
          &ScriptModuleSerializer::writeFiles,
          py::arg("code_dir") = ".data/ts_code/code/")
      .def(
          "storage_context",
          &ScriptModuleSerializer::storage_context,
          pybind11::return_value_policy::reference_internal);

  // Used by torch.package to coordinate sharing of storages between eager
  // and ScriptModules.
  py::class_<
      SerializationStorageContext,
      std::shared_ptr<SerializationStorageContext>>(
      m, "SerializationStorageContext")
      .def("has_storage", &SerializationStorageContext::hasStorage)
      .def("get_or_add_storage", &SerializationStorageContext::getOrAddStorage);

  // torch.jit.ScriptModule is a subclass of this C++ object.
  // Methods here are prefixed with _ since they should not be
  // public.
  py::class_<Module, Object>(m, "ScriptModule")
      .def(py::init<std::string, std::shared_ptr<CompilationUnit>, bool>())
      .def(
          "save",
          [](Module& m,
             const std::string& filename,
             const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
            m.save(filename, _extra_files);
          },
          py::arg("filename"),
          py::arg("_extra_files") = ExtraFilesMap())
      .def(
          "save_to_buffer",
          [](Module& m, const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
            std::ostringstream buf;
            m.save(buf, _extra_files);
            return py::bytes(buf.str());
          },
          py::arg("_extra_files") = ExtraFilesMap())
      .def(
          "_save_for_mobile",
          [](Module& m,
             const std::string& filename,
             const ExtraFilesMap& _extra_files = ExtraFilesMap(),
             bool _save_mobile_debug_info = false,
             bool _use_flatbuffer = false) {
            m._save_for_mobile(
                filename,
                _extra_files,
                _save_mobile_debug_info,
                _use_flatbuffer);
          },
          py::arg("filename"),
          py::arg("_extra_files") = ExtraFilesMap(),
          py::arg("_save_mobile_debug_info") = false,
          py::arg("_use_flatbuffer") = false)
      .def(
          "_save_to_buffer_for_mobile",
          [](Module& m,
             const ExtraFilesMap& _extra_files = ExtraFilesMap(),
             bool _save_mobile_debug_info = false,
             bool _use_flatbuffer = false) {
            std::ostringstream buf;
            m._save_for_mobile(
                buf, _extra_files, _save_mobile_debug_info, _use_flatbuffer);
            return py::bytes(buf.str());
          },
          py::arg("_extra_files") = ExtraFilesMap(),
          py::arg("_save_mobile_debug_info") = false,
          py::arg("_use_flatbuffer") = false)
      .def("_set_optimized", &Module::set_optimized)
      .def(
          "dump",
          &Module::dump,
          py::arg("code") = true,
          py::arg("attrs") = true,
          py::arg("params") = true)
      .def(
          "dump_to_str",
          &Module::dump_to_str,
          py::arg("code") = true,
          py::arg("attrs") = true,
          py::arg("params") = true)
      .def(
          "_replicate_for_data_parallel",
          [](Module& module) {
            const ModulePtr& obj = module._ivalue();
            auto copy = c10::ivalue::Object::create(
                c10::StrongTypePtr(obj->compilation_unit(), obj->type()),
                obj->slots().size());
            for (size_t i = 0; i < obj->slots().size(); ++i) {
              copy->setSlot(i, obj->getSlot(i));
            }
            return Module(std::move(copy));
          })
      .def(
          "get_debug_state",
          [](Module& self) {
            if (auto m = self.find_method("forward")) {
              return m->get_executor().getDebugState();
            }
            throw std::runtime_error(
                "Attempted to call get_debug_state on a Module without a compiled forward()");
          })
      .def(
          "_define",
          [](Module& m,
             std::shared_ptr<ConcreteModuleType> concreteType,
             const std::string& script,
             const ResolutionCallback& rcb) {
            const auto self = ModuleSelf(std::move(concreteType));
            m._ivalue()->compilation_unit()->define(
                *m.type()->name(), script, pythonResolver(rcb), &self);
            didFinishEmitModule(m);
          })
      .def(
          "_register_attribute",
          [](Module& m,
             const std::string& name,
             const TypePtr& type,
             py::handle value) {
            m.register_attribute(name, type, toIValue(value, type));
          })
      .def(
          "_create_method_from_trace",
          [](Module& self,
             const std::string& name,
             const py::function& func,
             const py::tuple& input_tuple,
             const py::function& var_name_lookup_fn,
             bool strict,
             bool force_outplace,
             const std::vector<std::string>& argument_names) {
            // prereq: Module's buffers and parameters are unique
            // this was ensured in python before calling this function
            auto typed_inputs = toTraceableStack(input_tuple);

            std::shared_ptr<Graph> graph =
                std::get<0>(tracer::createGraphByTracing(
                    func,
                    typed_inputs,
                    var_name_lookup_fn,
                    strict,
                    force_outplace,
                    &self,
                    argument_names));
            const auto method_name = QualifiedName(*self.type()->name(), name);
            auto fn = self._ivalue()->compilation_unit()->create_function(
                method_name, graph);
            self.type()->addMethod(fn);
            didFinishEmitModule(self);
          },
          py::arg("name"),
          py::arg("func"),
          py::arg("input_tuple"),
          py::arg("var_name_lookup_fn"),
          py::arg("strict"),
          py::arg("force_outplace"),
          py::arg("argument_names") = std::vector<std::string>())
      .def(
          "_create_method_from_trace_with_dict",
          [](Module& self,
             const std::string& name,
             const py::function& func,
             const py::dict& input_dict,
             const py::function& var_name_lookup_fn,
             bool strict,
             bool force_outplace,
             const std::vector<std::string>& argument_names) {
            // prereq: Module's buffers and parameters are unique
            // this was ensured in python before calling this function
            auto typed_inputs = toTraceableStack(input_dict);

            std::shared_ptr<Graph> graph =
                std::get<0>(tracer::createGraphByTracingWithDict(
                    func,
                    input_dict,
                    typed_inputs,
                    var_name_lookup_fn,
                    strict,
                    force_outplace,
                    &self,
                    argument_names));
            const auto method_name = QualifiedName(*self.type()->name(), name);
            auto fn = self._ivalue()->compilation_unit()->create_function(
                method_name, graph);
            self.type()->addMethod(fn);
            didFinishEmitModule(self);
          },
          py::arg("name"),
          py::arg("func"),
          py::arg("input_dict"),
          py::arg("var_name_lookup_fn"),
          py::arg("strict"),
          py::arg("force_outplace"),
          py::arg("argument_names") = std::vector<std::string>())
      .def(
          "_get_forward_hooks",
          [](const Module& m) {
            std::vector<StrongFunctionPtr> funcs;
            for (auto& hook : m.type()->getForwardHooks()) {
              funcs.emplace_back(
                  StrongFunctionPtr(m.type()->compilation_unit(), hook));
            }
            return funcs;
          })
      .def(
          "_get_forward_pre_hooks",
          [](const Module& m) {
            std::vector<StrongFunctionPtr> funcs;
            for (auto& pre_hook : m.type()->getForwardPreHooks()) {
              funcs.emplace_back(
                  StrongFunctionPtr(m.type()->compilation_unit(), pre_hook));
            }
            return funcs;
          })
      .def_property_readonly(
          "code",
          [](Module& self) {
            std::vector<at::IValue> constants;
            PrintDepsTable deps;
            PythonPrint pp(constants, deps);
            pp.printNamedType(self.type());
            return pp.str();
          })
      .def_property_readonly(
          "code_with_constants",
          [](Module& self) {
            std::vector<at::IValue> constants;
            PrintDepsTable deps;
            PythonPrint pp(constants, deps);
            pp.printNamedType(self.type());
            std::map<std::string, at::IValue> consts;
            int i = 0;
            for (auto const& constant : constants) {
              consts["c" + std::to_string(i)] = constant;
              i += 1;
            }
            return std::make_tuple(pp.str(), consts);
          })
      .def("apply", &Module::apply)
      .def("__copy__", &Module::copy)
      .def(
          "__hash__",
          [](const Module& self) {
            // Similar to Tensor's `__hash__`, which is `id()`.
            return std::hash<c10::ivalue::Object*>{}(self._ivalue().get());
          })
      .def(
          "__eq__",
          [](const Module& self, const py::object& other) {
            // TODO: call UDF if it exists
            if (!py::isinstance<Module>(other)) {
              return false;
            }
            return self._ivalue().get() ==
                py::cast<Module>(other)._ivalue().get();
          })
      .def(
          "__deepcopy__",
          [](const Module& self, const py::dict& memo) {
            return Module(
                pyIValueDeepcopy(IValue(self._ivalue()), memo).toObject());
          })
      .def("children", &Module::children)
      .def_property_readonly("qualified_name", [](const Module& self) {
        return self.type()->name()->qualifiedName();
      });

  py::class_<mobile::Module>(m, "LiteScriptModule")
      .def(py::init<
           c10::intrusive_ptr<c10::ivalue::Object>,
           std::shared_ptr<mobile::CompilationUnit>>())
      .def(
          "find_method",
          [](mobile::Module& m, const std::string& method_name) {
            auto method = m.find_method(method_name);
            return method != c10::nullopt;
          },
          py::arg("method_name"))
      .def(
          "run_method",
          [](mobile::Module& m,
             const std::string& method_name,
             const py::tuple& input_tuple) {
            Stack stack;
            for (auto& input : input_tuple) {
              stack.push_back(toTypeInferredIValue(input));
            }
            return m.get_method(method_name)(stack);
          },
          py::arg("method_name"),
          py::arg("input_tuple"))
      .def(
          "forward",
          [](mobile::Module& m, const py::tuple& input_tuple) {
            Stack stack;
            for (auto& input : input_tuple) {
              stack.push_back(toTypeInferredIValue(input));
            }
            return m.get_method("forward")(stack);
          },
          py::arg("input_tuple"));

  slot_dict_impl<detail::ParameterPolicy>::bind(m, "ParameterDict");
  slot_dict_impl<detail::BufferPolicy>::bind(m, "BufferDict");
  slot_dict_impl<detail::ModulePolicy>::bind(m, "ModuleDict");

  py::class_<ErrorReport, std::shared_ptr<ErrorReport>>(m, "ErrorReport")
      .def(py::init<SourceRange>())
      .def("what", &ErrorReport::what)
      .def_static("call_stack", ErrorReport::current_call_stack);

  py::class_<CompilationUnit, std::shared_ptr<CompilationUnit>>(
      m, "CompilationUnit")
      .def(
          py::init([](const std::string& lang, const uint32_t _frames_up) {
            auto cu = std::make_shared<CompilationUnit>();
            if (lang.size() > 0) {
              pyCompilationUnitDefine(*cu, lang, nullptr, _frames_up);
            }
            return cu;
          }),
          py::arg("lang") = "",
          py::arg("_frames_up") = 0)

      .def(
          "find_function",
          [](std::shared_ptr<CompilationUnit> self, const std::string& name) {
            auto fn = self->find_function(QualifiedName(name));
            if (fn) {
              return c10::optional<StrongFunctionPtr>(
                  StrongFunctionPtr(std::move(self), fn));
            } else {
              return c10::optional<StrongFunctionPtr>(c10::nullopt);
            }
          })
      .def(
          "__getattr__",
          [](std::shared_ptr<CompilationUnit> self, const std::string& name) {
            auto fn = self->find_function(QualifiedName(name));
            if (fn) {
              return StrongFunctionPtr(std::move(self), fn);
            } else {
              throw AttributeError(
                  "'CompilationUnit' has no attribute '%s'", name.c_str());
            }
          })
      .def(
          "get_functions",
          [](const std::shared_ptr<CompilationUnit>& self) {
            auto raw_functions = self->get_functions();
            std::vector<StrongFunctionPtr> functions;
            functions.reserve(raw_functions.size());
            for (auto fn : raw_functions) {
              if (fn) {
                functions.emplace_back(self, fn);
              }
            }
            return functions;
          })
      .def("set_optimized", &CompilationUnit::set_optimized)
      .def(
          "define",
          pyCompilationUnitDefine,
          py::arg("src"),
          py::arg("rcb") = nullptr,
          py::arg("_frames_up") = 0)
      .def(
          "create_function",
          [](std::shared_ptr<CompilationUnit>& self,
             const std::string& qualified_name,
             std::shared_ptr<Graph> graph,
             bool should_mangle) {
            Function* fn = self->create_function(
                qualified_name, std::move(graph), should_mangle);
            return StrongFunctionPtr(std::move(self), fn);
          },
          py::arg("qualified_name"),
          py::arg("graph"),
          py::arg("should_mangle") = false)
      .def(
          "get_interface",
          [](const std::shared_ptr<CompilationUnit>& self,
             const std::string& name) { return self->get_interface(name); })
      .def(
          "get_class",
          [](const std::shared_ptr<CompilationUnit>& self,
             const std::string& name) { return self->get_class(name); })
      .def(
          "drop_all_functions",
          [](const std::shared_ptr<CompilationUnit>& self) {
            self->drop_all_functions();
          });

  py::class_<StrongFunctionPtr>(m, "ScriptFunction", py::dynamic_attr())
      .def(
          "__call__",
          [](py::args args, py::kwargs kwargs) {
            HANDLE_TH_ERRORS
            // see: [pybind11 varargs]
            auto strongPtr = py::cast<StrongFunctionPtr>(args[0]);
            Function& callee = *strongPtr.function_;
            py::object result = invokeScriptFunctionFromPython(
                callee,
                // NOLINTNEXTLINE(performance-move-const-arg)
                tuple_slice(std::move(args), 1),
                // NOLINTNEXTLINE(performance-move-const-arg)
                std::move(kwargs));
            return result;
            END_HANDLE_TH_ERRORS_PYBIND
          })
      .def(
          "save",
          [](const StrongFunctionPtr& self,
             const std::string& filename,
             const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
            Module module("__torch__.PlaceholderModule");
            // [issue 27343]
            // Modules have 'training' attributes by default, but due to
            // https://github.com/pytorch/pytorch/issues/27343, functions end
            // up having a training attribute when they are loaded. This adds
            // a fake 'training' attribute that shouldn't be used, but prevents
            // jitter on saving and loading. Once that issue is fixed this can
            // be deleted.
            module.register_attribute("training", BoolType::get(), true);
            addFunctionToModule(module, self);
            module.save(filename, _extra_files);
          },
          py::arg("filename"),
          py::arg("_extra_files") = ExtraFilesMap())
      .def(
          "save_to_buffer",
          [](const StrongFunctionPtr& self,
             const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
            std::ostringstream buf;
            Module module("__torch__.PlaceholderModule");
            // see [issue 27343]
            module.register_attribute("training", BoolType::get(), true);
            addFunctionToModule(module, self);
            module.save(buf, _extra_files);
            return py::bytes(buf.str());
          },
          py::arg("_extra_files") = ExtraFilesMap())
      .def_property_readonly(
          "graph",
          [](const StrongFunctionPtr& self) {
            return toGraphFunction(*self.function_).graph();
          })
      .def_property_readonly(
          "inlined_graph",
          [](const StrongFunctionPtr& self) {
            auto g = toGraphFunction(*self.function_).graph()->copy();
            Inline(*g);
            return g;
          })
      .def_property_readonly(
          "schema",
          [](const StrongFunctionPtr& self) {
            return self.function_->getSchema();
          })
      .def_property_readonly(
          "code",
          [](const StrongFunctionPtr& self) {
            std::vector<at::IValue> constants;
            PrintDepsTable deps;

            PythonPrint pp(constants, deps);
            pp.printFunction(*self.function_);
            return pp.str();
          })
      .def(
          "get_debug_state",
          [](const StrongFunctionPtr& self) {
            return toGraphFunction(*self.function_)
                .get_executor()
                .getDebugState();
          })
      .def(
          "_debug_flush_compilation_cache",
          [](const StrongFunctionPtr& self) {
            toGraphFunction(*self.function_)
                .get_executor()
                .debugFlushCompilationCache();
          })
      .def_property_readonly(
          "name",
          [](const StrongFunctionPtr& self) { return self.function_->name(); })
      .def(
          "_set_ignore_amp",
          [](StrongFunctionPtr& self, bool ignore) {
            auto fn = self.function_;
            TORCH_INTERNAL_ASSERT(fn->isGraphFunction());
            GraphFunction& g_fn = toGraphFunction(*fn);
            g_fn._set_ignore_amp(ignore);
          })
      .def_property_readonly(
          "qualified_name",
          [](const StrongFunctionPtr& self) {
            return self.function_->qualname().qualifiedName();
          })
      .def_property_readonly("__doc__", [](const StrongFunctionPtr& self) {
        return self.function_->doc_string();
      });

  py::class_<Method>(m, "ScriptMethod", py::dynamic_attr())
      .def(
          "__call__",
          [](py::args args, py::kwargs kwargs) {
            // see: [pybind11 varargs]
            HANDLE_TH_ERRORS
            Method& method = py::cast<Method&>(args[0]);

            return invokeScriptMethodFromPython(
                method,
                // NOLINTNEXTLINE(performance-move-const-arg)
                tuple_slice(std::move(args), 1),
                // NOLINTNEXTLINE(performance-move-const-arg)
                std::move(kwargs));
            END_HANDLE_TH_ERRORS_PYBIND
          })
      .def_property_readonly("graph", &Method::graph)
      .def_property_readonly(
          "inlined_graph",
          [](const Method& self) {
            auto g = toGraphFunction(self.function()).graph()->copy();
            Inline(*g);
            return g;
          })
      .def_property_readonly(
          "schema", [](Method& m) { return m.function().getSchema(); })
      .def_property_readonly("name", &Method::name)
      .def_property_readonly(
          "code",
          [](Method& self) {
            std::vector<at::IValue> constants;
            PrintDepsTable deps;
            PythonPrint pp(constants, deps);
            pp.printMethod(self.function());
            return pp.str();
          })
      .def(
          "_debug_flush_compilation_cache",
          [](Method& self) {
            return self.get_executor().debugFlushCompilationCache();
          })
      .def_property_readonly(
          "code_with_constants",
          [](Method& self) {
            std::vector<at::IValue> constants;
            PrintDepsTable deps;
            PythonPrint pp(constants, deps);
            pp.printMethod(self.function());
            std::map<std::string, at::IValue> consts;
            int i = 0;
            for (auto const& constant : constants) {
              consts["c" + std::to_string(i)] = constant;
              i += 1;
            }
            return std::make_tuple(pp.str(), consts);
          })
      .def_property_readonly("owner", &Method::owner);
  m.def("_generate_upgraders_graph", &generate_upgraders_graph);
  m.def(
      "_calculate_package_version_based_on_upgraders",
      &calculate_package_version_based_on_upgraders);
  m.def("_get_version_calculator_flag", &get_version_calculator_flag);
  m.def(
      "_compile_graph_to_code_table",
      [](const std::string& name, const std::shared_ptr<Graph>& graph) {
        CompilationOptions options;
        GraphFunction jitFunc(name, graph, nullptr);
        auto mobileFunc = convertJitFunctionToMobileFunction(jitFunc, options);
        return convertMobileFunctionToCodeTable(*mobileFunc, options);
      });
  m.def(
      "_jit_script_compile",
      [](const std::string& qualname,
         const Def& def,
         const ResolutionCallback& rcb,
         const FunctionDefaults& defaults) {
        C10_LOG_API_USAGE_ONCE("torch.script.compile");
        const auto name = c10::QualifiedName(qualname);
        TORCH_INTERNAL_ASSERT(name.name() == def.name().name());
        return script_compile_function(name, def, defaults, rcb);
      });
  m.def(
      "_jit_script_compile_overload",
      [](const std::string& qualname,
         const Decl& overload_decl,
         const Def& implementation_def,
         const ResolutionCallback& rcb,
         const FunctionDefaults& implementation_defaults,
         const py::object& signature) {
        const auto name = c10::QualifiedName(qualname);
        return script_compile_overloaded_function(
            name,
            overload_decl,
            implementation_def,
            rcb,
            implementation_defaults,
            signature);
      });
  m.def(
      "_replace_overloaded_method_decl",
      [](const Decl& overload_decl,
         const Def& implementation_def,
         const std::string& new_name) {
        checkOverloadDecl(overload_decl, implementation_def.decl());
        return implementation_def.withDecl(overload_decl).withName(new_name);
      });
  m.def(
      "_create_function_from_trace",
      [](const std::string& qualname,
         const py::function& func,
         const py::tuple& input_tuple,
         const py::function& var_name_lookup_fn,
         bool strict,
         bool force_outplace,
         const std::vector<std::string>& argument_names) {
        auto typed_inputs = toTraceableStack(input_tuple);
        std::shared_ptr<Graph> graph = std::get<0>(tracer::createGraphByTracing(
            func,
            typed_inputs,
            var_name_lookup_fn,
            strict,
            force_outplace,
            /*self=*/nullptr,
            argument_names));

        auto cu = get_python_cu();
        auto name = c10::QualifiedName(qualname);
        auto result = cu->create_function(
            std::move(name), std::move(graph), /*shouldMangle=*/true);
        StrongFunctionPtr ret(std::move(cu), result);
        didFinishEmitFunction(ret);
        return ret;
      },
      py::arg("name"),
      py::arg("func"),
      py::arg("input_tuple"),
      py::arg("var_name_lookup_fn"),
      py::arg("strict"),
      py::arg("force_outplace"),
      py::arg("argument_names") = std::vector<std::string>());

  m.def(
      "_create_function_from_trace_with_dict",
      [](const std::string& qualname,
         const py::function& func,
         const py::dict& input_dict,
         const py::function& var_name_lookup_fn,
         bool strict,
         bool force_outplace,
         const std::vector<std::string>& argument_names) {
        auto typed_inputs = toTraceableStack(input_dict);
        std::shared_ptr<Graph> graph =
            std::get<0>(tracer::createGraphByTracingWithDict(
                func,
                input_dict,
                typed_inputs,
                var_name_lookup_fn,
                strict,
                force_outplace,
                /*self=*/nullptr,
                argument_names));

        auto cu = get_python_cu();
        auto name = c10::QualifiedName(qualname);
        auto result = cu->create_function(
            std::move(name), std::move(graph), /*shouldMangle=*/true);
        StrongFunctionPtr ret(std::move(cu), result);
        didFinishEmitFunction(ret);
        return ret;
      },
      py::arg("name"),
      py::arg("func"),
      py::arg("input_dict"),
      py::arg("var_name_lookup_fn"),
      py::arg("strict"),
      py::arg("force_outplace"),
      py::arg("argument_names") = std::vector<std::string>());

  m.def(
      "_jit_script_class_compile",
      [](const std::string& qualifiedName,
         const ClassDef& classDef,
         const ClassMethodDefaults& defaults,
         const ResolutionCallback& rcb) {
        C10_LOG_API_USAGE_ONCE("torch.script.class");
        if (classDef.superclass().present()) {
          throw ErrorReport(classDef.range())
              << "Torchscript does not support class inheritance.";
        }
        auto cu = get_python_cu();
        auto classname = c10::QualifiedName(qualifiedName);
        if (cu->get_type(classname) != nullptr) {
          classname = cu->mangle(classname);
        }

        auto classType = ClassType::create(
            classname,
            cu,
            /* is_module = */ false,
            /* doc_string = */ "",
            getUnresolvedClassAttributes(classDef));
        cu->register_type(classType);
        std::vector<ResolverPtr> methodRcbs, propRcbs;
        std::vector<Def> methodDefs;
        std::vector<Property> props;

        for (const auto& def : classDef.body()) {
          if (def.kind() != TK_DEF) {
            throw ErrorReport(def.range())
                << "Currently class bodies can only contain method "
                   "definitions. File an issue on GitHub if you want "
                   "something else!";
          }
          methodDefs.emplace_back(Def(def));
          methodRcbs.push_back(
              pythonResolver(rcb, classDef.name().name(), classType));
        }

        // Gather definitions for property getters and setters as well as
        // corresponding resolution callbacks.
        if (classDef.properties().present()) {
          for (const auto& prop : classDef.properties().get()) {
            props.emplace_back(prop);
            propRcbs.push_back(
                pythonResolver(rcb, classDef.name().name(), classType));
          }
        }

        const auto self = SimpleSelf(classType);
        cu->define(classname, props, propRcbs, methodDefs, methodRcbs, &self);

        // Stitch in default arguments for methods. Properties don't need to be
        // considered since there is no way to invoke setters without passing in
        // a value.
        auto defs_it = methodDefs.begin();
        while (defs_it != methodDefs.end()) {
          auto def_name = (*defs_it).name().name();
          // If the method is not in the defaults map, assume there are
          // no default arguments for it.
          auto default_it = defaults.find(def_name);
          if (default_it == defaults.end()) {
            continue;
          }

          const auto method_name =
              QualifiedName(classname, (*defs_it).name().name());
          auto& method = cu->get_function(method_name);
          method.setSchema(getSchemaWithNameAndDefaults(
              defs_it->range(),
              method.getSchema(),
              at::nullopt,
              default_it->second));
          ++defs_it;
        }
        return classType;
      });
  m.def(
      "_jit_script_interface_compile",
      [](const std::string& qualifiedName,
         const ClassDef& classDef,
         const ResolutionCallback& rcb,
         bool is_module) {
        auto cu = get_python_cu();
        auto className = c10::QualifiedName(qualifiedName);
        if (cu->get_type(className) != nullptr) {
          className = cu->mangle(className);
        }

        get_python_cu()->define_interface(
            className, classDef, pythonResolver(rcb), is_module);
        return className.qualifiedName();
      });

  py::class_<torch::jit::ErrorReport::CallStack>(
      m, "CallStack", py::dynamic_attr())
      .def(py::init<const std::string&, const SourceRange&>());

  m.def("_parse_source_def", [](const std::string& src) {
    Parser p(std::make_shared<Source>(src));
    return Def(p.parseFunction(/*is_method=*/true));
  });
  m.def("parse_type_comment", [](const std::string& comment) {
    Parser p(std::make_shared<Source>(comment));
    return Decl(p.parseTypeComment());
  });

  m.def("_get_upgraders_map_size", &get_upgraders_map_size);
  m.def("_dump_upgraders_map", &dump_upgraders_map);

  m.def("_test_only_populate_upgraders", &test_only_populate_upgraders);
  m.def("_test_only_remove_upgraders", &test_only_remove_upgraders);

  m.def("merge_type_from_type_comment", &mergeTypesFromTypeComment);
  m.def("_get_max_operator_version", &getMaxOperatorVersion);
  m.def("_get_operator_version_map", &get_operator_version_map);
  m.def("_get_upgrader_ranges", &getUpgradersRangeForOp);
  m.def("_test_only_add_entry_to_op_version_map", &test_only_add_entry);
  m.def("_test_only_remove_entry_to_op_version_map", &test_only_remove_entry);
  m.def(
      "import_ir_module",
      [](std::shared_ptr<CompilationUnit> cu,
         const std::string& filename,
         py::object map_location,
         const py::dict& extra_files) {
        c10::optional<at::Device> optional_device;
        if (!map_location.is_none()) {
          AT_ASSERT(THPDevice_Check(map_location.ptr()));
          optional_device =
              reinterpret_cast<THPDevice*>(map_location.ptr())->device;
        }
        ExtraFilesMap extra_files_map = extra_files_from_python(extra_files);
        auto ret = import_ir_module(
            std::move(cu), filename, optional_device, extra_files_map);
        extra_files_to_python(extra_files_map, extra_files);
        return ret;
      });
  m.def(
      "_import_ir_module_from_package",
      [](std::shared_ptr<CompilationUnit> cu,
         std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader,
         std::shared_ptr<torch::jit::DeserializationStorageContext>
             storage_context,
         py::object map_location,
         std::string ts_id) {
        c10::optional<at::Device> optional_device;
        if (!map_location.is_none()) {
          AT_ASSERT(THPDevice_Check(map_location.ptr()));
          optional_device =
              reinterpret_cast<THPDevice*>(map_location.ptr())->device;
        }
        return import_ir_module(
            std::move(cu),
            std::move(reader),
            std::move(storage_context),
            optional_device,
            std::move(ts_id));
      });
  m.def(
      "import_ir_module_from_buffer",
      [](std::shared_ptr<CompilationUnit> cu,
         const std::string& buffer,
         py::object map_location,
         const py::dict& extra_files) {
        std::istringstream in(buffer);
        c10::optional<at::Device> optional_device;
        if (!map_location.is_none()) {
          AT_ASSERT(THPDevice_Check(map_location.ptr()));
          optional_device =
              reinterpret_cast<THPDevice*>(map_location.ptr())->device;
        }
        ExtraFilesMap extra_files_map = extra_files_from_python(extra_files);
        auto ret = import_ir_module(
            std::move(cu), in, optional_device, extra_files_map);
        extra_files_to_python(extra_files_map, extra_files);
        return ret;
      });
  m.def(
      "_load_for_lite_interpreter",
      [](const std::string& filename, py::object map_location) {
        c10::optional<at::Device> optional_device;
        if (!map_location.is_none()) {
          AT_ASSERT(THPDevice_Check(map_location.ptr()));
          optional_device =
              reinterpret_cast<THPDevice*>(map_location.ptr())->device;
        }
        return _load_for_mobile(filename, optional_device);
      });
  m.def(
      "_load_for_lite_interpreter_from_buffer",
      [](const std::string& buffer, py::object map_location) {
        std::istringstream in(buffer);
        c10::optional<at::Device> optional_device;
        if (!map_location.is_none()) {
          AT_ASSERT(THPDevice_Check(map_location.ptr()));
          optional_device =
              reinterpret_cast<THPDevice*>(map_location.ptr())->device;
        }
        return _load_for_mobile(in, optional_device);
      });
  m.def(
      "_backport_for_mobile",
      [](const std::string& filename_input,
         const std::string& filename_output,
         const int64_t version) {
        return _backport_for_mobile(filename_input, filename_output, version);
      });
  m.def(
      "_backport_for_mobile_from_buffer",
      [](const std::string& buffer_input,
         const std::string& filename_output,
         const int64_t version) {
        std::istringstream in(buffer_input);
        return _backport_for_mobile(in, filename_output, version);
      });
  m.def(
      "_backport_for_mobile_to_buffer",
      [](const std::string& filename_input, const int64_t version) {
        std::ostringstream buffer_output;
        bool success =
            _backport_for_mobile(filename_input, buffer_output, version);
        return success ? py::bytes(buffer_output.str()) : py::bytes("");
      });
  m.def(
      "_backport_for_mobile_from_buffer_to_buffer",
      [](const std::string& buffer_input, const int64_t version) {
        std::istringstream in(buffer_input);
        std::ostringstream buffer_output;
        bool success = _backport_for_mobile(in, buffer_output, version);
        return success ? py::bytes(buffer_output.str()) : py::bytes("");
      });
  m.def("_get_model_bytecode_version", [](const std::string& filename) {
    return _get_model_bytecode_version(filename);
  });
  m.def(
      "_get_model_extra_files",
      [](const std::string& filename, const py::dict& py_extra_files) {
        c10::optional<at::Device> optional_device;
        ExtraFilesMap cpp_extra_files = ExtraFilesMap();
        _load_for_mobile(filename, optional_device, cpp_extra_files);
        extra_files_to_python(cpp_extra_files, py_extra_files);

        return py_extra_files;
      });
  m.def(
      "_get_model_bytecode_version_from_buffer", [](const std::string& buffer) {
        std::istringstream in(buffer);
        return _get_model_bytecode_version(in);
      });
  m.def(
      "_get_model_extra_files_from_buffer",
      [](const std::string& buffer, const py::dict& py_extra_files) {
        c10::optional<at::Device> optional_device;
        ExtraFilesMap cpp_extra_files = ExtraFilesMap();
        std::istringstream in(buffer);
        _load_for_mobile(in, optional_device, cpp_extra_files);
        extra_files_to_python(cpp_extra_files, py_extra_files);

        return py_extra_files;
      });
  m.def("_get_mobile_model_contained_types", [](const std::string& filename) {
    return _get_mobile_model_contained_types(filename);
  });
  m.def(
      "_get_mobile_model_contained_types_from_buffer",
      [](const std::string& buffer) {
        std::istringstream in(buffer);
        return _get_mobile_model_contained_types(in);
      });
  m.def("_nn_module_to_mobile", [](const Module& module) {
    CompilationOptions options;
    return jitModuleToMobile(module, options);
  });
  py::class_<OperatorInfo>(m, "OperatorInfo")
      .def_readonly("num_schema_args", &OperatorInfo::num_schema_args);
  m.def("_get_model_ops_and_info", [](const std::string& filename) {
    return _get_model_ops_and_info(filename);
  });
  m.def("_get_model_ops_and_info_from_buffer", [](const std::string& buffer) {
    std::istringstream in(buffer);
    return _get_model_ops_and_info(in);
  });
  m.def("_export_operator_list", [](torch::jit::mobile::Module& sm) {
    return debugMakeSet(torch::jit::mobile::_export_operator_list(sm));
  });
  m.def(
      "_quantize_ondevice_ptq_dynamic",
      [](mobile::Module& m, const std::string& method_name) {
        mobile::quantization::PTQQuanizationHelper ptq_helper;
        ptq_helper.quantize_dynamic(m, method_name);
      });

  m.def("_jit_set_emit_hooks", setEmitHooks);
  m.def("_jit_get_emit_hooks", getEmitHooks);
  m.def("_jit_clear_class_registry", []() {
    get_python_cu()->_clear_python_cu();
  });
  m.def(
      "_debug_set_autodiff_subgraph_inlining",
      debugSetAutodiffSubgraphInlining);
  m.def("_debug_set_fusion_group_inlining", debugSetFusionGroupInlining);
  m.def("_debug_get_fusion_group_inlining", getFusionGroupInlining);
  m.def("_propagate_shapes", _propagate_shapes);
  m.def(
      "_propagate_and_assign_input_shapes", _propagate_and_assign_input_shapes);
  m.def(
      "_last_executed_optimized_graph",
      []() { return lastExecutedOptimizedGraph(); },
      "Retrieve the optimized graph that was run the last time the graph executor ran on this thread");
  m.def(
      "_create_function_from_graph",
      [](const std::string& qualname, std::shared_ptr<Graph> graph) {
        // TODO this should go in the global Python CU
        auto cu = std::make_shared<CompilationUnit>();
        c10::QualifiedName name(qualname);
        auto fn = cu->create_function(std::move(name), std::move(graph));
        return StrongFunctionPtr(std::move(cu), fn);
      });
  m.def("_ivalue_tags_match", ivalue_tags_match);
  m.def("_ivalue_debug_python_object", [](py::object py_obj) {
    // convert to IValue first, IValue will incref via py::object
    IValue pyobj_ivalue = toIValue(std::move(py_obj), PyObjectType::get());
    // convert back to PyObject by borrowing the reference, which also
    // incref, after the return of this function, IValue is out of scope
    // which decref, so the return value is original refcount + 1
    py::object ret = toPyObject(pyobj_ivalue);
    return ret;
  });
  m.def("_jit_debug_module_iterators", _jit_debug_module_iterators);

  py::class_<testing::FileCheck>(m, "FileCheck")
      .def(py::init<>())
      .def("check", &testing::FileCheck::check)
      .def("check_not", &testing::FileCheck::check_not)
      .def("check_same", &testing::FileCheck::check_same)
      .def("check_next", &testing::FileCheck::check_next)
      .def("check_count", &testing::FileCheck::check_count)
      .def("check_dag", &testing::FileCheck::check_dag)
      .def(
          "check_source_highlighted",
          &testing::FileCheck::check_source_highlighted)
      .def(
          "check_count",
          [](testing::FileCheck& f,
             const std::string& str,
             size_t count,
             bool exactly) { return f.check_count(str, count, exactly); },
          "Check Count",
          py::arg("str"),
          py::arg("count"),
          py::arg("exactly") = false)
      .def(
          "run",
          [](testing::FileCheck& f, const std::string& str) {
            return f.run(str);
          })
      .def(
          "run", [](testing::FileCheck& f, const Graph& g) { return f.run(g); })
      .def(
          "run",
          [](testing::FileCheck& f,
             const std::string& input,
             const std::string& output) { return f.run(input, output); },
          "Run",
          py::arg("checks_file"),
          py::arg("test_file"))
      .def(
          "run",
          [](testing::FileCheck& f, const std::string& input, const Graph& g) {
            return f.run(input, g);
          },
          "Run",
          py::arg("checks_file"),
          py::arg("graph"));

  m.def(
      "_logging_set_logger",
      [](logging::LoggerBase* logger) { return logging::setLogger(logger); },
      py::return_value_policy::reference);
  m.def("_set_graph_executor_optimize", [](bool optimize) {
    setGraphExecutorOptimize(optimize);
  });

  m.def(
      "_get_graph_executor_optimize",
      [](c10::optional<bool> new_setting = c10::nullopt) {
        bool old_value = getGraphExecutorOptimize();
        if (new_setting) {
          setGraphExecutorOptimize(*new_setting);
        }
        return old_value;
      },
      py::arg("new_settings") = nullptr);

  m.def(
      "_enable_mobile_interface_call_export",
      &torch::jit::enableMobileInterfaceCallExport);

  m.def("_create_module_with_type", [](const ClassTypePtr& type) {
     return Module(get_python_cu(), type);
   }).def("_create_object_with_type", [](const ClassTypePtr& type) {
    return Object(get_python_cu(), type);
  });

  m.def("_export_opnames", [](Module& sm) {
    return debugMakeList(torch::jit::export_opnames(sm));
  });

  py::class_<
      ConcreteModuleTypeBuilder,
      std::shared_ptr<ConcreteModuleTypeBuilder>>(
      m, "ConcreteModuleTypeBuilder")
      .def(py::init<py::object>())
      .def(
          "add_constant",
          [](ConcreteModuleTypeBuilder& self,
             std::string name,
             py::object value) {
            self.addConstant(std::move(name), std::move(value));
          })
      .def("add_attribute", &ConcreteModuleTypeBuilder::addAttribute)
      .def(
          "add_function_attribute",
          &ConcreteModuleTypeBuilder::addFunctionAttribute)
      .def(
          "add_builtin_function",
          &ConcreteModuleTypeBuilder::addBuiltinFunction)
      .def("add_forward_hook", &ConcreteModuleTypeBuilder::addForwardHook)
      .def(
          "add_forward_pre_hook", &ConcreteModuleTypeBuilder::addForwardPreHook)
      .def("add_module", &ConcreteModuleTypeBuilder::addModule)
      .def("add_overload", &ConcreteModuleTypeBuilder::addOverload)
      .def("set_poisoned", &ConcreteModuleTypeBuilder::setPoisoned)
      .def(
          "add_failed_attribute",
          &ConcreteModuleTypeBuilder::addFailedAttribute)
      .def(
          "add_ignored_attribute",
          &ConcreteModuleTypeBuilder::addIgnoredAttribute)
      .def(
          "add_ignored_attributes",
          [](ConcreteModuleTypeBuilder& self,
             const std::vector<std::string>& names) {
            for (auto& name : names) {
              self.addIgnoredAttribute(name);
            }
          })
      .def(
          "set_module_dict",
          [](ConcreteModuleTypeBuilder& self) {
            self.setIterableModuleKind(IterableModuleKind::DICT);
          })
      .def("build", &ConcreteModuleTypeBuilder::build)
      .def(
          "equals",
          [](const ConcreteModuleTypeBuilder& self,
             const ConcreteModuleTypeBuilder& other) {
            return self.equals(other);
          })
      .def(
          "set_module_list",
          [](ConcreteModuleTypeBuilder& self) {
            self.setIterableModuleKind(IterableModuleKind::LIST);
          })
      .def(
          "set_parameter_list",
          [](ConcreteModuleTypeBuilder& self) {
            self.setIterableModuleKind(IterableModuleKind::PARAMLIST);
          })
      .def("set_parameter_dict", [](ConcreteModuleTypeBuilder& self) {
        self.setIterableModuleKind(IterableModuleKind::PARAMDICT);
      });

  py::class_<ConcreteModuleType, std::shared_ptr<ConcreteModuleType>>(
      m, "ConcreteModuleType")
      .def_property_readonly("py_class", &ConcreteModuleType::getPyClass)
      .def_property_readonly("jit_type", &ConcreteModuleType::getJitType)
      .def_static("from_jit_type", &ConcreteModuleType::fromJitType)
      .def("get_constants", &ConcreteModuleType::getConstantsPy)
      .def("get_attributes", &ConcreteModuleType::getAttributesPy)
      .def("get_modules", &ConcreteModuleType::getModulesPy)
      .def("dump", &ConcreteModuleType::dump)
      .def("is_ignored_attribute", &ConcreteModuleType::isIgnoredAttribute)
      .def(
          "equals",
          [](const ConcreteModuleType& self, const ConcreteModuleType& other) {
            return self.equals(other);
          })
      .def(
          "equals",
          [](const ConcreteModuleType& self,
             const ConcreteModuleTypeBuilder& other) {
            return self.equals(other);
          })
      .def(
          "_create_methods_and_properties",
          [](std::shared_ptr<ConcreteModuleType> concreteType,
             const std::vector<Property>& properties,
             const std::vector<ResolutionCallback>& propertyRcbs,
             const std::vector<Def>& methodDefs,
             const std::vector<ResolutionCallback>& methodRcbs,
             const std::vector<FunctionDefaults>& defaults) {
            TORCH_INTERNAL_ASSERT(methodDefs.size() == methodRcbs.size());
            TORCH_INTERNAL_ASSERT(properties.size() == propertyRcbs.size());

            std::vector<ResolverPtr> methodResolvers, propertyResolvers;
            methodResolvers.reserve(methodRcbs.size());
            for (auto& callback : methodRcbs) {
              methodResolvers.push_back(pythonResolver(callback));
            }

            propertyResolvers.reserve(propertyRcbs.size());
            for (auto& callback : propertyRcbs) {
              propertyResolvers.push_back(pythonResolver(callback));
            }

            const auto& selfType =
                concreteType->getJitType()->expect<ClassType>();
            const auto& prefix = selfType->name().value();
            const auto self = ModuleSelf(std::move(concreteType));
            auto cu = selfType->compilation_unit();
            cu->define(
                prefix,
                properties,
                propertyResolvers,
                methodDefs,
                methodResolvers,
                &self);
            // Stitch in default arguments for each Def if provided
            auto defaults_it = defaults.begin();
            auto defs_it = methodDefs.begin();
            while (defs_it != methodDefs.end()) {
              const auto method_name =
                  QualifiedName(prefix, (*defs_it).name().name());
              auto& method = cu->get_function(method_name);
              method.setSchema(getSchemaWithNameAndDefaults(
                  defs_it->range(),
                  method.getSchema(),
                  at::nullopt,
                  *defaults_it));
              ++defs_it;
              ++defaults_it;
            }
          })
      .def(
          "_create_hooks",
          [](std::shared_ptr<ConcreteModuleType> concreteType,
             const std::vector<Def>& hookDefs,
             const std::vector<ResolutionCallback>& hookRcbs,
             const std::vector<Def>& preHookDefs,
             const std::vector<ResolutionCallback>& preHookRcbs) {
            TORCH_INTERNAL_ASSERT(hookDefs.size() == hookRcbs.size());
            TORCH_INTERNAL_ASSERT(preHookDefs.size() == preHookRcbs.size());

            std::vector<ResolverPtr> hookResolvers, preHookResolvers;

            hookResolvers.reserve(hookRcbs.size());
            for (auto& callback : hookRcbs) {
              hookResolvers.push_back(pythonResolver(callback));
            }

            preHookResolvers.reserve(preHookRcbs.size());
            for (auto& callback : preHookRcbs) {
              preHookResolvers.push_back(pythonResolver(callback));
            }

            const auto& selfType =
                concreteType->getJitType()->expect<ClassType>();
            const auto& prefix = selfType->name().value();
            const auto self = ModuleSelf(std::move(concreteType));
            auto cu = selfType->compilation_unit();
            cu->define_hooks(
                prefix,
                hookDefs,
                hookResolvers,
                preHookDefs,
                preHookResolvers,
                &self);
          });

  m.def(
      "_resolve_type",
      [](const std::string& name,
         const SourceRange& range,
         const ResolutionCallback& rcb) {
        return pythonResolver(rcb)->resolveType(name, range);
      });
  m.def(
      "_resolve_type_from_object",
      [](const py::object& obj,
         const SourceRange& range,
         const ResolutionCallback& rcb) {
        return pythonResolver(rcb)->resolveTypeFromObject(obj, range);
      });

  m.def(
      "_run_emit_module_hook", [](const Module& m) { didFinishEmitModule(m); });

  m.def(
      "_set_should_use_format_with_string_table",
      setShouldUseFormatWithStringTable);

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<logging::LoggerBase, std::shared_ptr<logging::LoggerBase>>(
      m, "LoggerBase");
  py::enum_<logging::LockingLogger::AggregationType>(m, "AggregationType")
      .value("SUM", logging::LockingLogger::AggregationType::SUM)
      .value("AVG", logging::LockingLogger::AggregationType::AVG)
      .export_values();
  py::class_<
      logging::LockingLogger,
      logging::LoggerBase,
      std::shared_ptr<logging::LockingLogger>>(m, "LockingLogger")
      .def(py::init<>())
      .def("set_aggregation_type", &logging::LockingLogger::setAggregationType)
      .def("get_counter_val", &logging::LockingLogger::getCounterValue);
  py::class_<
      logging::NoopLogger,
      logging::LoggerBase,
      std::shared_ptr<logging::NoopLogger>>(m, "NoopLogger")
      .def(py::init<>());
  m.def("_jit_is_script_object", [](const py::object& obj) {
    return py::isinstance<Object>(obj);
  });

  m.def("_get_file_format", [](const std::string& path) {
    switch (getFileFormat(path)) {
      case FileFormat::FlatbufferFileFormat:
        return "flatbuffer";
      case FileFormat::ZipFileFormat:
        return "zipfile";
      default:
        return "invalid";
    }
  });

  m.def(
      "_save_parameters",
      [](const std::map<std::string, at::Tensor>& map,
         const std::string& filename,
         bool use_flatbuffer = false) {
        _save_parameters(map, filename, use_flatbuffer);
      });

  initScriptDictBindings(module);
  initScriptListBindings(module);
}
} // namespace jit
} // namespace torch
