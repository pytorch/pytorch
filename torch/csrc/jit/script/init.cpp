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
  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      Function& m,
      const SourceRange& loc) const override {
    AutoGIL ag;
    py::object obj = rcb_(name);
    if (obj.is(py::none())) {
      return nullptr;
    }
    return toSugaredValue(obj, m, loc);
  }

  TypePtr resolveType(const std::string& name) const override {
    AutoGIL ag;
    py::object obj = rcb_(name);
    if (obj.is(py::none())) {
      return nullptr;
    }
    py::bool_ isClass = py::module::import("inspect").attr("isclass")(obj);
    if (!py::cast<bool>(isClass)) {
      return nullptr;
    }

    py::str qualifiedName =
        py::module::import("torch.jit").attr("_qualified_name")(obj);

    return ClassType::get(c10::QualifiedName(qualifiedName));
  }

 private:
  ResolutionCallback rcb_;
};

std::shared_ptr<PythonResolver> pythonResolver(ResolutionCallback rcb) {
  return std::make_shared<PythonResolver>(rcb);
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
            << "Expected a default value of type " << arg.type()->str()
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

static Self moduleSelf(
    const std::shared_ptr<Module>& m,
    const py::object& py_m) {
  return [m, py_m](Value* v) {
    v->setType(m->module_object()->type());
    return std::make_shared<ModuleValue>(v, m, py_m);
  };
}

static void setInputTensorTypes(Graph& g, const Stack& stack) {
  AT_ASSERT(stack.size() == g.inputs().size());
  for (size_t i = 0; i < stack.size(); ++i) {
    g.inputs().at(i)->setType(
        DimensionedTensorType::create(stack.at(i).toTensor()));
  }
}

static std::shared_ptr<Graph> _propagate_shapes(
    Graph& graph,
    std::vector<at::Tensor> inputs,
    bool with_grad = false) {
  Stack stack(inputs.begin(), inputs.end());
  auto retval = graph.copy();
  setInputTensorTypes(*retval, stack);
  PropagateInputShapes(retval);
  return retval;
}

static std::shared_ptr<Graph> _propagate_and_assign_input_and_output_shapes(
    Graph& graph,
    std::vector<at::Tensor> inputs,
    std::vector<at::Tensor> outputs,
    bool with_grad = false,
    bool propagate = true) {
  auto retval = graph.copy();
  if (propagate) {
    setInputTensorTypes(*retval, fmap<IValue>(inputs));
    PropagateInputShapes(retval);
  }
  AT_ASSERT(retval->inputs().size() == inputs.size());
  for (size_t i = 0; i < retval->inputs().size(); ++i) {
    auto scalar_type = inputs[i].scalar_type();
    auto sizes = inputs[i].sizes();
    auto type =
        torch::jit::CompleteTensorType::create(scalar_type, at::kCPU, sizes);
    retval->inputs()[i]->setType(type);
  }
  at::ArrayRef<Value*> output_values = retval->outputs();
  // patch this to still work if we are returning a tuple of multiple values
  if (output_values.at(0)->type()->kind() == TupleType::Kind) {
    AT_ASSERT(output_values.at(0)->node()->kind() == prim::TupleConstruct);
    output_values = output_values.at(0)->node()->inputs();
  }
  AT_ASSERT(output_values.size() == outputs.size());
  for (size_t i = 0; i < retval->outputs().size(); ++i) {
    auto scalar_type = outputs[i].scalar_type();
    auto sizes = outputs[i].sizes();
    auto type =
        torch::jit::CompleteTensorType::create(scalar_type, at::kCPU, sizes);
    output_values[i]->setType(type);
  }
  return retval;
}

void initJitScriptBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // STL containers are not mutable by default and hence we need to bind as
  // follows.
  py::bind_map<ExtraFilesMap>(m, "ExtraFilesMap");

  // torch.jit.ScriptModule is a subclass of this C++ object.
  // Methods here are prefixed with _ since they should not be
  // public.
  py::class_<Module, std::shared_ptr<Module>>(m, "ScriptModule")
      .def(py::init<>())
      .def(
          "save",
          [](std::shared_ptr<Module> m,
             const std::string& filename,
             const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
            m->save(filename, _extra_files);
          },
          py::arg("filename"),
          py::arg("_extra_files") = ExtraFilesMap())
      .def(
          "save_to_buffer",
          [](std::shared_ptr<Module> m,
             const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
            std::ostringstream buf;
            m->save(buf, _extra_files);
            return py::bytes(buf.str());
          },
          py::arg("_extra_files") = ExtraFilesMap())
      .def("_set_optimized", &Module::set_optimized)
      .def(
          "_define",
          [](std::shared_ptr<Module> m,
             py::object py_m,
             const std::string& script,
             ResolutionCallback rcb) {
            c10::optional<Self> self;
            m->class_compilation_unit().define(
                script, pythonResolver(rcb), moduleSelf(m, py_m));
            didFinishEmitModule(m);
          })
      .def(
          "_create_methods",
          [](std::shared_ptr<Module> m,
             py::object py_m,
             const std::vector<Def>& defs,
             const std::vector<ResolutionCallback>& rcbs,
             const std::vector<FunctionDefaults>& defaults) {
            std::vector<ResolverPtr> resolvers;
            resolvers.reserve(rcbs.size());
            for (auto& callback : rcbs) {
              resolvers.push_back(pythonResolver(callback));
            }
            m->class_compilation_unit().define(
                defs, resolvers, moduleSelf(m, py_m));
            // Stitch in default arguments for each Def if provided
            auto defaults_it = defaults.begin();
            auto defs_it = defs.begin();
            while (defs_it != defs.end()) {
              auto& method = m->class_compilation_unit().get_function(
                  (*defs_it).name().name());
              method.setSchema(getSchemaWithNameAndDefaults(
                  defs_it->range(),
                  method.getSchema(),
                  at::nullopt,
                  *defaults_it));
              ++defs_it;
              ++defaults_it;
            }
            didFinishEmitModule(m);
          })
      .def(
          "_get_method",
          [](Module& self, const std::string& name) -> const Method& {
            return self.get_method(name);
          },
          py::return_value_policy::reference_internal)
      .def("_register_parameter", &Module::register_parameter)
      .def(
          "_register_attribute",
          [](Module& self, std::string name, TypePtr type, py::object value) {
            self.register_attribute(name, type, toIValue(value, type));
          })
      .def("_register_module", &Module::register_module)
      .def("_register_buffer", &Module::register_buffer)
      .def("_set_parameter", &Module::set_parameter)
      .def("_get_parameter", &Module::get_parameter)
      .def("_get_buffer", &Module::get_buffer)
      .def("_get_attribute", &Module::get_attribute)
      .def("_get_module", &Module::get_module)
      .def(
          "_get_modules",
          [](Module& self) -> py::tuple {
            auto modules = self.get_modules();
            py::tuple result(modules.size());
            for (size_t i = 0; i < modules.size(); ++i) {
              auto& item = modules[i];
              result[i] = std::make_pair(item->name(), item);
            }
            return result;
          })
      .def(
          "_get_parameters",
          [](Module& self) -> py::tuple {
            auto parameters = self.get_parameters();
            py::tuple result(parameters.size());
            for (size_t i = 0; i < parameters.size(); ++i) {
              auto& p = parameters[i];
              py::tuple r(2);
              result[i] = std::make_tuple(
                  p.name(), autograd::as_variable_ref(p.value().toTensor()));
            }
            return result;
          })
      .def(
          "_get_attributes",
          [](Module& self) -> py::tuple {
            auto attributes = self.get_attributes();
            py::tuple result(attributes.size());
            for (size_t i = 0; i < attributes.size(); ++i) {
              auto& buffer = attributes[i];
              py::tuple r(3);
              IValue v = buffer.value();
              result[i] = std::make_tuple(
                  buffer.name(), buffer.type(), toPyObject(std::move(v)));
            }
            return result;
          })
      .def(
          "_has_attribute",
          [](Module& self, const std::string& name) -> bool {
            return self.find_attribute(name);
          })
      .def(
          "_has_parameter",
          [](Module& self, const std::string& name) -> bool {
            return self.find_parameter(name);
          })
      .def(
          "_has_buffer",
          [](Module& self, const std::string& name) -> bool {
            return self.find_buffer(name);
          })
      .def(
          "_has_module",
          [](Module& self, const std::string& name) {
            return bool(self.find_module(name));
          })
      .def(
          "_has_method",
          [](Module& self, const std::string& name) {
            return bool(self.find_method(name));
          })
      .def(
          "_method_names",
          [](Module& self) {
            return fmap(
                self.get_methods(), [](const std::unique_ptr<Method>& method) {
                  return method->name();
                });
          })
      .def(
          "_create_method_from_trace",
          [](std::shared_ptr<Module> self,
             const std::string& name,
             py::function func,
             py::tuple input_tuple,
             py::function var_lookup_fn,
             bool force_outplace) {
            // prereq: Module's buffers and parameters are unique
            // this was ensured in python before calling this function
            auto typed_inputs = toTypedStack(input_tuple);
            auto graph = tracer::createGraphByTracing(
                func, typed_inputs, var_lookup_fn, force_outplace, self);
            self->module_object()->type()->compilation_unit().create_function(
                name, graph);
            didFinishEmitModule(self);
          })
      .def(
          "get_debug_state",
          [](Module& self) {
            if (self.find_method("forward")) {
              Method& m = self.get_method("forward");
              return m.get_executor().getDebugState();
            }
            throw std::runtime_error(
                "Attempted to call get_debug_state on a Module without a compiled forward()");
          })
      .def_property_readonly(
          "code",
          [](Module& self) {
            std::ostringstream ss;
            std::vector<at::Tensor> tensors;
            std::vector<ClassTypePtr> classes;
            PythonPrint(
                ss,
                self.class_compilation_unit(),
                true,
                tensors,
                classes,
                false);
            return ss.str();
          })
      .def("apply", &Module::apply)
      .def("_copy_into", &Module::copy_into)
      .def(
          "clone_method",
          [](std::shared_ptr<Module> m,
             std::shared_ptr<Module> orig,
             const std::string& name) { m->clone_method(*orig, name); });

  py::class_<CompilationUnit, std::shared_ptr<CompilationUnit>>(
      m, "CompilationUnit")
      .def(py::init<>())
      .def("find_function", &CompilationUnit::find_function)
      .def("set_optimized", &CompilationUnit::set_optimized)
      .def(
          "define",
          [](CompilationUnit& cu,
             const std::string& src,
             ResolutionCallback rcb) {
            cu.define(src, pythonResolver(rcb), nullptr);
          });

  py::class_<Function, std::shared_ptr<Function>>(
      m, "Function", py::dynamic_attr())
      .def(
          "__call__",
          [](py::args args, py::kwargs kwargs) {
            // see: [pybind11 varargs]
            Function& callee = py::cast<Function&>(args[0]);
            bool tracing = tracer::isTracing();
            if (tracing) {
              tracer::getTracingState()->graph->push_scope(callee.name());
            }
            py::object result = invokeScriptMethodFromPython(
                callee, tuple_slice(std::move(args), 1), std::move(kwargs));
            if (tracing) {
              tracer::getTracingState()->graph->pop_scope();
            }
            return result;
          })
      .def_property_readonly("graph", &Function::graph)
      .def_property_readonly("schema", &Function::getSchema)
      .def_property_readonly(
          "code",
          [](Function& self) {
            std::ostringstream ss;
            std::vector<at::Tensor> tensors;
            std::vector<ClassTypePtr> classes;
            PythonPrint(ss, self, false, tensors, classes, false);
            return ss.str();
          })
      .def(
          "get_debug_state",
          [](Function& self) { return self.get_executor().getDebugState(); })
      .def_property_readonly("name", &Function::name);

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
      .def(
          "initial_ivalues",
          [](Method& m) {
            std::vector<at::Tensor> tensors;
            for (auto& t : m.initial_ivalues()) {
              tensors.push_back(t.value().toTensor());
            }
            return tensors;
          })
      .def_property_readonly("schema", &Method::getSchema)
      .def_property_readonly("code", [](Method& self) {
        std::ostringstream ss;
        std::vector<at::Tensor> tensors;
        std::vector<ClassTypePtr> classes;
        PythonPrint(ss, self.function(), true, tensors, classes, false);
        return ss.str();
      });

  m.def(
      "_jit_script_compile",
      [](const Def& def, ResolutionCallback rcb, FunctionDefaults defaults) {
        CompilationUnit cu;
        cu.define({def}, {pythonResolver(rcb)}, nullptr);
        std::shared_ptr<Function> defined = cu.get_functions().at(0);
        defined->setSchema(getSchemaWithNameAndDefaults(
            def.range(), defined->getSchema(), def.name().name(), defaults));
        didFinishEmitFunction(defined);
        return defined;
      });

  m.def(
      "_create_function_from_trace",
      [](std::string name,
         py::function func,
         py::tuple input_tuple,
         py::function var_lookup_fn,
         bool force_outplace) {
        auto typed_inputs = toTypedStack(input_tuple);
        auto graph = tracer::createGraphByTracing(
            func, typed_inputs, var_lookup_fn, force_outplace);
        CompilationUnit cu;
        auto result = cu.create_function(std::move(name), std::move(graph));
        didFinishEmitFunction(result);
        return result;
      });

  m.def(
      "_jit_script_class_compile",
      [](const ClassDef& classDef, ResolutionCallback rcb) {
        auto cu = std::make_shared<CompilationUnit>();
        auto classType =
            ClassType::create(c10::QualifiedName(classDef.name().name()), cu);
        std::vector<ResolverPtr> rcbs;
        std::vector<Def> methodDefs;
        for (const auto& def : classDef.defs()) {
          methodDefs.push_back(def);
          rcbs.push_back(pythonResolver(rcb));
        }
        cu->define(methodDefs, rcbs, simpleSelf(classType));
      });

  m.def("parse_type_comment", [](const std::string& comment) {
    Parser p(comment);
    return Decl(p.parseTypeComment());
  });

  m.def("merge_type_from_type_comment", &mergeTypesFromTypeComment);
  m.def(
      "import_ir_module",
      [](ModuleLookup module_lookup,
         const std::string& filename,
         py::object map_location,
         ExtraFilesMap& extra_files) {
        c10::optional<at::Device> optional_device;
        if (!map_location.is(py::none())) {
          AT_ASSERT(THPDevice_Check(map_location.ptr()));
          optional_device =
              reinterpret_cast<THPDevice*>(map_location.ptr())->device;
        }
        import_ir_module(module_lookup, filename, optional_device, extra_files);
      });
  m.def(
      "import_ir_module_from_buffer",
      [](ModuleLookup module_lookup,
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
        import_ir_module(module_lookup, in, optional_device, extra_files);
      });

  m.def(
      "_jit_import_functions",
      [](CompilationUnit& cu,
         const std::string& src,
         const std::vector<at::Tensor>& constant_table,
         const Self& self) {
        import_functions(cu, src, constant_table, self, nullptr);
      });

  m.def("_jit_set_emit_hooks", setEmitHooks);
  m.def("_jit_clear_class_registry", ClassType::clearRegistry);
  m.def(
      "_debug_set_autodiff_subgraph_inlining",
      debugSetAutodiffSubgraphInlining);
  m.def("_propagate_shapes", _propagate_shapes);
  m.def(
      "_propagate_and_assign_input_and_output_shapes",
      _propagate_and_assign_input_and_output_shapes);
  m.def("_jit_python_print", [](py::object obj) {
    std::ostringstream ss;
    std::vector<at::Tensor> constants;
    std::vector<ClassTypePtr> classes;
    if (auto self = as_module(obj)) {
      PythonPrint(
          ss, self->class_compilation_unit(), true, constants, classes, true);
    } else if (auto self = as_function(obj)) {
      PythonPrint(ss, *self, false, constants, classes, true);
    } else {
      auto& m = py::cast<Method&>(obj);
      PythonPrint(ss, m.function(), true, constants, classes, true);
    }
    return std::make_pair(ss.str(), std::move(constants));
  });
  m.def(
      "_last_executed_optimized_graph",
      []() { return lastExecutedOptimizedGraph(); },
      "Retrieve the optimized graph that was run the last time the graph executor ran on this thread");
  m.def(
      "_create_function_from_graph",
      [](const std::string& name, std::shared_ptr<Graph> graph) {
        return CompilationUnit().create_function(name, graph);
      });

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
