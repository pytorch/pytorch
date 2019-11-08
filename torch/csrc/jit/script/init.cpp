#include <torch/csrc/jit/script/init.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/jit/import.h>
#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/script/module_python.h>
#include <torch/csrc/jit/script/python_sugared_value.h>
#include <torch/csrc/jit/script/sugared_value.h>
#include <torch/csrc/jit/testing/file_check.h>

#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/hooks_for_testing.h>
#include <torch/csrc/jit/import_source.h>
#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/passes/python_print.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/jit/python_tracer.h>
#include <torch/csrc/jit/script/logging.h>
#include <torch/csrc/jit/script/parser.h>
#include <torch/csrc/jit/tracer.h>

#include <torch/csrc/api/include/torch/ordered_dict.h>

#include <ATen/ATen.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/qualified_name.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <chrono>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

PYBIND11_MAKE_OPAQUE(torch::jit::script::ExtraFilesMap);

namespace torch {
namespace jit {
namespace script {

using ::c10::Argument;
using ::c10::FunctionSchema;

using ResolutionCallback = std::function<py::function(std::string)>;
using FunctionDefaults = std::unordered_map<std::string, py::object>;

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
      Function& m,
      const SourceRange& loc) override {
    AutoGIL ag;
    py::object obj = rcb_(name);
    if (obj.is(py::none())) {
      return nullptr;
    }
    return toSugaredValue(obj, m, loc);
  }

  static bool isNamedTupleClass(py::object obj) {
    auto tuple_type = reinterpret_cast<PyObject*>(&PyTuple_Type);
    return PyObject_IsSubclass(obj.ptr(), tuple_type) &&
        py::hasattr(obj, "_fields");
  }

  TypePtr resolveType(const std::string& name, const SourceRange& loc)
      override {
    if (classType_ && name == classname_) {
      return classType_;
    }
    AutoGIL ag;
    py::object obj = rcb_(name);
    if (obj.is(py::none())) {
      return nullptr;
    }
    py::bool_ isClass = py::module::import("inspect").attr("isclass")(obj);
    if (!py::cast<bool>(isClass)) {
      return nullptr;
    }

    auto qualifiedName = c10::QualifiedName(py::cast<std::string>(
        py::module::import("torch.jit").attr("_qualified_name")(obj)));

    if (isNamedTupleClass(obj)) {
      // Currently don't support default values
      if (py::hasattr(obj, "_field_defaults")) {
        auto default_dict = py::cast<std::map<std::string, py::object>>(
            py::getattr(obj, "_field_defaults"));
        if (default_dict.size()) {
          std::string error_msg =
              "Default values are currently not supported"
              " on NamedTuple fields in TorchScript. Fields "
              "with default values: [";
          bool first = true;
          for (const auto& kv : default_dict) {
            if (!first) {
              error_msg += ", ";
            }
            error_msg += kv.first;
          }
          error_msg += "]";
          throw ErrorReport(loc) << error_msg;
        }
      }

      py::object props = py::module::import("torch.jit")
                             .attr("_get_named_tuple_properties")(obj);
      std::string unqualName;
      std::vector<std::string> fields;
      std::vector<TypePtr> annotations;
      std::tie(unqualName, fields, annotations) = py::cast<
          std::tuple<std::string, decltype(fields), decltype(annotations)>>(
          props);

      auto tt = TupleType::createNamed(qualifiedName, fields, annotations);
      if (auto type = get_python_cu()->get_type(qualifiedName)) {
        TORCH_CHECK(
            type->isSubtypeOf(tt),
            "Can't to redefine NamedTuple: ",
            tt->python_str());
            return type;
      }
      get_python_cu()->register_type(tt);
      return tt;
    }
    return get_python_cu()->get_type(qualifiedName);
  }

 private:
  ResolutionCallback rcb_;
  std::string classname_;
  ClassTypePtr classType_;
};

std::shared_ptr<PythonResolver> pythonResolver(ResolutionCallback rcb) {
  return std::make_shared<PythonResolver>(rcb);
}
std::shared_ptr<PythonResolver> pythonResolver(
    ResolutionCallback rcb,
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
  for (size_t i = 0; i < new_decl.params().size(); ++i) {
    TORCH_INTERNAL_ASSERT(
        new_params[i].ident().name() == old_params[i].ident().name(),
        "Overload parameters must have the same names\n",
        new_params[i].ident(),
        old_params[i].ident());
  }
}
} // namespace

FunctionSchema getSchemaWithNameAndDefaults(
    const SourceRange& range,
    const FunctionSchema& schema,
    const at::optional<std::string>& new_name,
    const FunctionDefaults& default_args) {
  std::vector<Argument> new_args;
  for (auto& arg : schema.arguments()) {
    auto it = default_args.find(arg.name());
    if (it != default_args.end()) {
      try {
        IValue value;
        auto n = arg.N();
        auto list_type = arg.type()->cast<ListType>();
        if (n && *n > 0 && list_type) {
          // BroadcastingList, allow default values T for arg types List[T]
          value = toIValue(it->second, list_type->getElementType());
        } else {
          value = toIValue(it->second, arg.type());
        }
        new_args.emplace_back(
            arg.name(), arg.type(), arg.N(), value, arg.kwarg_only());
      } catch (py::cast_error& e) {
        throw ErrorReport(range)
            << "Expected a default value of type " << arg.type()->python_str()
            << " on parameter \"" << arg.name() << "\"";
      }
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

static StrongFunctionPtr script_compile_function(
    const c10::QualifiedName& name,
    const Def& def,
    const FunctionDefaults& defaults,
    ResolutionCallback rcb) {
  auto cu = get_python_cu();
  auto defined_functions = cu->define(
      QualifiedName(name.prefix()),
      {def},
      {pythonResolver(std::move(rcb))},
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
    return concreteType_->getJitType();
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

static TupleTypePtr getTupleTensorType(
    const Stack::const_iterator& s_iter,
    const Stack::const_iterator& s_iter_end,
    const TypePtr& tupleType,
    bool complete) {
  AT_ASSERT(tupleType->kind() == TupleType::Kind);
  AT_ASSERT(s_iter != s_iter_end);

  std::vector<TypePtr> types;
  for (const auto& subType : tupleType->containedTypes()) {
    if (subType->kind() == TupleType::Kind) {
      types.push_back(
          getTupleTensorType(s_iter + 1, s_iter_end, subType, complete));
    } else {
      types.push_back(getTensorType(s_iter->toTensor(), complete));
    }
  }
  return TupleType::create(types);
}

static void setInputTensorTypes(Graph& g, const Stack& stack, bool complete) {
  at::ArrayRef<Value*> input_values = g.inputs();
  auto s_iter = stack.begin();
  for (auto v : input_values) {
    AT_ASSERT(s_iter != stack.end());
    if (v->type()->kind() == TupleType::Kind) {
      AT_ASSERT(v->node()->kind() == prim::Param);
      v->setType(getTupleTensorType(s_iter, stack.end(), v->type(), complete));
    } else {
      v->setType(getTensorType(s_iter->toTensor(), complete));
      s_iter++;
    }
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
    bool with_grad = false,
    bool propagate = true) {
  auto retval = graph.copy();
  setInputTensorTypes(*retval, fmap<IValue>(inputs), /*complete=*/true);
  if (propagate) {
    PropagateInputShapes(retval);
  }
  return retval;
}

static std::shared_ptr<Graph> _assign_output_shapes(
    Graph& graph,
    std::vector<at::Tensor> outputs) {
  auto retval = graph.copy();
  AT_ASSERT(retval->outputs().size() == outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto scalar_type = outputs[i].scalar_type();
    auto sizes = outputs[i].sizes();
    auto type =
        torch::jit::TensorType::createContiguous(scalar_type, at::kCPU, sizes);
    retval->outputs()[i]->setType(type);
  }
  return retval;
}

void addFunctionToModule(Module& module, const StrongFunctionPtr& func) {
  // Make a graph with a fake self argument
  auto graph = func.function_->graph()->copy();
  auto v = graph->insertInput(0, "self");
  v->setType(module.module_object()->type());
  const auto name = QualifiedName(module.name(), "forward");
  auto method = module.class_compilation_unit()->create_function(name, graph);
  module.type()->addMethod(method);
}

// this is used in our test suite to check that we correctly preserved type tags
bool ivalue_tags_match(const Module& lhs, const Module& rhs) {
  struct Work {
    IValue a;
    IValue b;
  };
  std::unordered_set<const void*> visited;
  std::vector<Work> work = {{lhs.module_object(), rhs.module_object()}};
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
    if (*unshapedType(item.a.type()) != *unshapedType(item.b.type())) {
      return false;
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
    } else if (item.a.isGenericList()) {
      auto al = item.a.toGenericList();
      auto bl = item.b.toGenericList();
      for (size_t i = 0; i < al.size(); ++i) {
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
  slot_dict_impl(script::ModulePtr module) : module_(std::move(module)) {}
  bool contains(const std::string& name) const {
    if (auto slot = module_->type()->findAttributeSlot(name)) {
      if (Policy::valid(module_->type(), *slot)) {
        return true;
      }
    }
    return false;
  }

  std::vector<std::pair<std::string, py::object>> items() const {
    std::vector<std::pair<std::string, py::object>> result;
    for (size_t i = 0, N = module_->type()->numAttributes(); i < N; ++i) {
      if (Policy::valid(module_->type(), i)) {
        result.emplace_back(
            module_->type()->getAttributeName(i),
            toPyObject(module_->getSlot(i)));
      }
    }
    return result;
  }

  void setattr(const std::string& name, py::object value) {
    const TypePtr& type = module_->type()->getAttribute(name);
    script::Module(module_).setattr(name, toIValue(std::move(value), type));
  }

  py::object getattr(const std::string& name) {
    return toPyObject(script::Module(module_).attr(name));
  }

  static void bind(const py::module& m, const char* name) {
    py::class_<slot_dict_impl<Policy>>(m, name)
        .def(py::init(
            [](Module& m) { return slot_dict_impl<Policy>(m.module_object()); }))
        .def("contains", &slot_dict_impl<Policy>::contains)
        .def("items", &slot_dict_impl<Policy>::items)
        .def("setattr", &slot_dict_impl<Policy>::setattr)
        .def("getattr", &slot_dict_impl<Policy>::getattr);
  }
 private:
  script::ModulePtr module_;
};

template <typename T>
py::list debugMakeList(const T& list) {
  py::list result;
  for (auto elem : list) {
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


void initJitScriptBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // STL containers are not mutable by default and hence we need to bind as
  // follows.
  py::bind_map<ExtraFilesMap>(m, "ExtraFilesMap");

  // torch.jit.ScriptModule is a subclass of this C++ object.
  // Methods here are prefixed with _ since they should not be
  // public.
  py::class_<Module>(m, "ScriptModule")
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
          py::arg("params") = true,
          py::arg("indent") = 0)
      .def(
          "_define",
          [](Module& m,
             std::shared_ptr<ConcreteModuleType> concreteType,
             const std::string& script,
             ResolutionCallback rcb) {
            const auto self = ModuleSelf(std::move(concreteType));
            m.class_compilation_unit()->define(
                m.name(), script, pythonResolver(rcb), &self);
            didFinishEmitModule(m);
          })
      .def("_type", [](Module& m) { return m.type(); })
      .def(
          "_get_method",
          [](Module& self, const std::string& name) -> Method {
            return self.get_method(name);
          },
          py::keep_alive<0, 1>())
      .def("_register_parameter", &Module::register_parameter)
      .def(
          "_register_attribute",
          [](Module& self, std::string name, TypePtr type, py::object value) {
            auto unshaped = unshapedType(type);
            self.register_attribute(
                name, unshaped, toIValue(std::move(value), type));
          })
      .def("_register_module", &Module::register_module)
      .def(
          "setattr",
          [](Module& self, const std::string& name, py::object value) {
            TypePtr type = self.type()->getAttribute(name);
            TORCH_CHECK(type, "Module has no attribute '", name, "'");
            auto ivalue = toIValue(std::move(value), type);
            self.setattr(name, ivalue);
          })
      .def(
          "getattr",
          [](Module& self, const std::string& name) {
            return toPyObject(self.attr(name));
          })
      .def(
          "hasattr",
          [](Module& self, const std::string& name) {
            return self.hasattr(name);
          })
      .def(
          "_replicate_for_data_parallel",
          [](Module& module) {
            Module replica(
                *module.module_object()->type()->name(),
                module.module_object()->compilation_unit(),
                /*should_mangle*/ true);
            ClassTypePtr module_cls = module.module_object()->type();
            for (size_t i = 0, N = module_cls->numAttributes(); i < N; ++i) {
              if (!detail::ModulePolicy::valid(module_cls, i) &&
                  !detail::ParameterPolicy::valid(module_cls, i) &&
                  !detail::BufferPolicy::valid(module_cls, i)) {
                replica.register_attribute(
                    module_cls->getAttributeName(i),
                    module_cls->getAttribute(i),
                    module.module_object()->getSlot(i));
              }
            }
            return replica;
          })
      .def(
          "_has_method",
          [](Module& self, const std::string& name) {
            return bool(self.find_method(name));
          })
      .def(
          "_method_names",
          [](Module& self) {
            return fmap(self.get_methods(), [](const Method& method) {
              return method.name();
            });
          })
      .def(
          "_create_method_from_trace",
          [](Module& self,
             const std::string& name,
             py::function func,
             py::tuple input_tuple,
             py::function var_lookup_fn,
             bool force_outplace) {
            // prereq: Module's buffers and parameters are unique
            // this was ensured in python before calling this function
            auto typed_inputs = toTraceableStack(input_tuple);

            std::shared_ptr<Graph> graph = std::get<0>(tracer::createGraphByTracing(
                func, typed_inputs, var_lookup_fn, force_outplace, &self));
            const auto method_name = QualifiedName(self.name(), name);
            auto fn = self.class_compilation_unit()->create_function(
                method_name, graph);
            self.type()->addMethod(fn);
            didFinishEmitModule(self);
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
      .def_property_readonly(
          "code",
          [](Module& self) {
            std::vector<at::Tensor> tensors;
            std::vector<c10::NamedTypePtr> deps;
            PythonPrint pp(tensors, deps, false);
            pp.printNamedType(self.type());
            return pp.str();
          })
      .def("apply", &Module::apply)
      .def("_clone", &Module::clone)
      .def_property_readonly(
          "name", [](const Module& self) { return self.name().name(); })
      .def(
          "clone_method", [](Module& m, Module& orig, const std::string& name) {
            m.clone_method(orig, name);
          });

  slot_dict_impl<script::detail::ParameterPolicy>::bind(m, "ParameterDict");
  slot_dict_impl<script::detail::BufferPolicy>::bind(m, "BufferDict");
  slot_dict_impl<script::detail::ModulePolicy>::bind(m, "ModuleDict");

  py::class_<ErrorReport, std::shared_ptr<ErrorReport>>(m, "ErrorReport")
      .def(py::init<SourceRange>())
      .def("what", &ErrorReport::what);

  py::class_<CompilationUnit, std::shared_ptr<CompilationUnit>>(
      m, "CompilationUnit")
      .def(py::init<>())
      .def(
          "find_function",
          [](std::shared_ptr<CompilationUnit> self, const std::string& name) {
            auto& fn = self->get_function(QualifiedName(name));
            return StrongFunctionPtr(std::move(self), &fn);
          })
      .def("set_optimized", &CompilationUnit::set_optimized)
      .def(
          "define",
          [](CompilationUnit& cu,
             const std::string& src,
             ResolutionCallback rcb) {
            cu.define(c10::nullopt, src, pythonResolver(rcb), nullptr);
          });

  py::class_<StrongFunctionPtr>(m, "ScriptFunction", py::dynamic_attr())
      .def(
          "__call__",
          [](py::args args, py::kwargs kwargs) {
            HANDLE_TH_ERRORS
            // see: [pybind11 varargs]
            auto strongPtr = py::cast<StrongFunctionPtr>(args[0]);
            Function& callee = *strongPtr.function_;
            bool tracing = tracer::isTracing();
            py::object result = invokeScriptFunctionFromPython(
                callee, tuple_slice(std::move(args), 1), std::move(kwargs));
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
            // Modules have 'training' attributes by defualt, but due to
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
          [](const StrongFunctionPtr& self) { return self.function_->graph(); })
      .def_property_readonly(
          "schema",
          [](const StrongFunctionPtr& self) {
            return self.function_->getSchema();
          })
      .def_property_readonly(
          "code",
          [](const StrongFunctionPtr& self) {
            std::vector<at::Tensor> tensors;
            std::vector<c10::NamedTypePtr> deps;
            PythonPrint pp(tensors, deps, false);
            pp.printFunction(*self.function_);
            return pp.str();
          })
      .def(
          "get_debug_state",
          [](const StrongFunctionPtr& self) {
            return self.function_->get_executor().getDebugState();
          })
      .def_property_readonly(
          "name",
          [](const StrongFunctionPtr& self) { return self.function_->name(); })
      .def_property_readonly(
          "qualified_name", [](const StrongFunctionPtr& self) {
            return self.function_->qualname().qualifiedName();
          });

  py::class_<Method>(m, "ScriptMethod", py::dynamic_attr())
      .def(
          "__call__",
          [](py::args args, py::kwargs kwargs) {
            // see: [pybind11 varargs]
            Method& method = py::cast<Method&>(args[0]);
            return invokeScriptMethodFromPython(
                method, tuple_slice(std::move(args), 1), std::move(kwargs));
          })
      .def_property_readonly("graph", &Method::graph)
      .def_property_readonly(
          "schema", [](Method& m) { return m.function().getSchema(); })
      .def_property_readonly("name", &Method::name)
      .def_property_readonly("code", [](Method& self) {
        std::vector<at::Tensor> tensors;
        std::vector<c10::NamedTypePtr> deps;
        PythonPrint pp(tensors, deps, false);
        pp.printMethod(self.function());
        return pp.str();
      });
  m.def(
      "_jit_script_compile",
      [](const std::string& qualname,
         const Def& def,
         ResolutionCallback rcb,
         const FunctionDefaults& defaults) {
        C10_LOG_API_USAGE_ONCE("torch.script.compile");
        const auto name = c10::QualifiedName(qualname);
        TORCH_INTERNAL_ASSERT(name.name() == def.name().name());
        return script_compile_function(name, def, defaults, std::move(rcb));
      });
  m.def(
      "_jit_script_compile_overload",
      [](const std::string& qualname,
         const Decl& overload_decl,
         const Def& implementation_def,
         ResolutionCallback rcb,
         const FunctionDefaults& defaults) {
        const auto name = c10::QualifiedName(qualname);
        checkOverloadDecl(overload_decl, implementation_def.decl());
        auto new_def = implementation_def.withDecl(overload_decl);
        return script_compile_function(name, new_def, defaults, std::move(rcb));
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
      [](std::string qualname,
         py::function func,
         py::tuple input_tuple,
         py::function var_lookup_fn,
         bool force_outplace) {
        auto typed_inputs = toTraceableStack(input_tuple);
        std::shared_ptr<Graph> graph = std::get<0>(tracer::createGraphByTracing(
            func, typed_inputs, var_lookup_fn, force_outplace));
        auto cu = get_python_cu();
        auto name = c10::QualifiedName(qualname);
        auto result = cu->create_function(
            std::move(name), std::move(graph), /*shouldMangle=*/true);
        StrongFunctionPtr ret(std::move(cu), result);
        didFinishEmitFunction(ret);
        return ret;
      });

  m.def(
      "_jit_script_class_compile",
      [](const std::string& qualifiedName,
         const ClassDef& classDef,
         ResolutionCallback rcb) {
        C10_LOG_API_USAGE_ONCE("torch.script.class");
        if (classDef.superclass().present()) {
          throw ErrorReport(classDef.range())
              << "Torchscript does not support class inheritance.";
        }
        auto cu = get_python_cu();
        const auto classname = c10::QualifiedName(qualifiedName);
        auto classType = ClassType::create(classname, cu);
        cu->register_type(classType);
        std::vector<ResolverPtr> rcbs;
        std::vector<Def> methodDefs;
        for (const auto& def : classDef.body()) {
          if (def.kind() != TK_DEF) {
            throw ErrorReport(def.range())
                << "Currently class bodies can only contain method "
                   "definitions. File an issue on Github if you want "
                   "something else!";
          }
          methodDefs.emplace_back(Def(def));
          rcbs.push_back(
              pythonResolver(rcb, classDef.name().name(), classType));
        }
        const auto self = SimpleSelf(classType);
        cu->define(classname, methodDefs, rcbs, &self);
      });
  m.def(
      "_jit_script_interface_compile",
      [](const std::string& qualifiedName,
         const ClassDef& classDef,
         ResolutionCallback rcb,
         bool is_module) {
        get_python_cu()->define_interface(
            c10::QualifiedName(qualifiedName),
            classDef,
            pythonResolver(std::move(rcb)),
            is_module);
      });

  m.def("_parse_source_def", [](const std::string& src) {
    Parser p(std::make_shared<Source>(src));
    return Def(p.parseFunction(/*is_method=*/true));
  });
  m.def("parse_type_comment", [](const std::string& comment) {
    Parser p(std::make_shared<Source>(comment));
    return Decl(p.parseTypeComment());
  });

  m.def("merge_type_from_type_comment", &mergeTypesFromTypeComment);
  m.def(
      "import_ir_module",
      [](std::shared_ptr<CompilationUnit> cu,
         const std::string& filename,
         py::object map_location,
         ExtraFilesMap& extra_files) {
        c10::optional<at::Device> optional_device;
        if (!map_location.is(py::none())) {
          AT_ASSERT(THPDevice_Check(map_location.ptr()));
          optional_device =
              reinterpret_cast<THPDevice*>(map_location.ptr())->device;
        }
        return import_ir_module(
            std::move(cu), filename, optional_device, extra_files);
      });
  m.def(
      "import_ir_module_from_buffer",
      [](std::shared_ptr<CompilationUnit> cu,
         const std::string& buffer,
         py::object map_location,
         ExtraFilesMap& extra_files) {
        std::istringstream in(buffer);
        c10::optional<at::Device> optional_device;
        if (!map_location.is(py::none())) {
          AT_ASSERT(THPDevice_Check(map_location.ptr()));
          optional_device =
              reinterpret_cast<THPDevice*>(map_location.ptr())->device;
        }
        return import_ir_module(
            std::move(cu), in, optional_device, extra_files);
      });

  m.def("_jit_set_emit_hooks", setEmitHooks);
  m.def("_jit_get_emit_hooks", getEmitHooks);
  m.def("_jit_clear_class_registry", []() {
    get_python_cu()->_clear_python_cu();
  });
  m.def(
      "_debug_set_autodiff_subgraph_inlining",
      debugSetAutodiffSubgraphInlining);
  m.def("_propagate_shapes", _propagate_shapes);
  m.def(
      "_propagate_and_assign_input_shapes",
      _propagate_and_assign_input_shapes);
  m.def("_assign_output_shapes", _assign_output_shapes);
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
        auto fn = cu->create_function(std::move(name), graph);
        return StrongFunctionPtr(std::move(cu), fn);
      });
  m.def("_ivalue_tags_match", ivalue_tags_match);
  m.def("_jit_debug_module_iterators", _jit_debug_module_iterators);

  py::class_<testing::FileCheck>(m, "FileCheck")
      .def(py::init<>())
      .def("check", &testing::FileCheck::check)
      .def("check_not", &testing::FileCheck::check_not)
      .def("check_same", &testing::FileCheck::check_same)
      .def("check_next", &testing::FileCheck::check_next)
      .def("check_count", &testing::FileCheck::check_count)
      .def("check_dag", &testing::FileCheck::check_dag)
      .def("check_count", &testing::FileCheck::check_count)
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

  m.def("_get_graph_executor_optimize", &torch::jit::getGraphExecutorOptimize);

  m.def("_create_module_with_type", [](const ClassTypePtr& type) {
    return Module(get_python_cu(), type);
  });

  py::class_<ConcreteModuleType, std::shared_ptr<ConcreteModuleType>>(
      m, "ConcreteModuleType")
      .def(py::init<>())
      .def_property_readonly("py_class", &ConcreteModuleType::getPyClass)
      .def_property_readonly("jit_type", &ConcreteModuleType::getJitType)
      .def("get_constants", &ConcreteModuleType::getConstantsPy)
      .def("get_attributes", &ConcreteModuleType::getAttributesPy)
      .def("get_modules", &ConcreteModuleType::getModulesPy)
      .def("add_constant", &ConcreteModuleType::addConstant)
      .def("add_attribute", &ConcreteModuleType::addAttribute)
      .def("add_function_attribute", &ConcreteModuleType::addFunctionAttribute)
      .def("add_module", &ConcreteModuleType::addModule)
      .def("add_module_interface", &ConcreteModuleType::addModuleInterface)
      .def("add_pyclass", &ConcreteModuleType::addPyClass)
      .def("add_overload", &ConcreteModuleType::addOverload)
      .def("add_jit_type", &ConcreteModuleType::addJitType)
      .def("set_poisoned", &ConcreteModuleType::setPoisoned)
      .def(
          "set_module_dict",
          [](ConcreteModuleType& self) {
            self.setIterableModuleKind(IterableModuleKind::DICT);
          })
      .def(
          "set_module_list",
          [](ConcreteModuleType& self) {
            self.setIterableModuleKind(IterableModuleKind::LIST);
          })
      .def(
          "create_new_type_from_this",
          &ConcreteModuleType::createNewTypeFromThis)
      .def("add_failed_attribute", &ConcreteModuleType::addFailedAttribute)
      .def("dump", &ConcreteModuleType::dump)
      .def(
          "equals",
          [](const ConcreteModuleType& self, const ConcreteModuleType& other) {
            return self == other;
          })
      .def(
          "_create_methods",
          [](std::shared_ptr<ConcreteModuleType> concreteType,
             const std::vector<Def>& defs,
             const std::vector<ResolutionCallback>& rcbs,
             const std::vector<FunctionDefaults>& defaults) {
            TORCH_INTERNAL_ASSERT(defs.size() == rcbs.size());
            std::vector<ResolverPtr> resolvers;
            resolvers.reserve(rcbs.size());
            for (auto& callback : rcbs) {
              resolvers.push_back(pythonResolver(callback));
            }
            const auto& selfType = concreteType->getJitType();
            const auto& prefix = selfType->name().value();
            const auto self = ModuleSelf(std::move(concreteType));
            auto cu = selfType->compilation_unit();
            cu->define(prefix, defs, resolvers, &self);
            // Stitch in default arguments for each Def if provided
            auto defaults_it = defaults.begin();
            auto defs_it = defs.begin();
            while (defs_it != defs.end()) {
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
          });

  m.def(
      "_resolve_type",
      [](const std::string& name, SourceRange range, ResolutionCallback rcb) {
        return pythonResolver(rcb)->resolveType(name, range);
      });

  m.def(
      "_run_emit_module_hook", [](const Module& m) { didFinishEmitModule(m); });

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
}
} // namespace script
} // namespace jit
} // namespace torch
