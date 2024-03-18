#include <torch/csrc/jit/python/python_custom_class.h>

#include <pybind11/embed.h>

#include <torch/csrc/jit/frontend/sugared_value.h>

#include <fmt/format.h>

namespace torch::jit {

struct CustomMethodProxy;
struct CustomObjectProxy;

py::object ScriptClass::__call__(py::args args, py::kwargs kwargs) {
  auto instance =
      Object(at::ivalue::Object::create(class_type_, /*numSlots=*/1));
  Function* init_fn = instance.type()->findMethod("__init__");
  TORCH_CHECK(
      init_fn,
      fmt::format(
          "Custom C++ class: '{}' does not have an '__init__' method bound. "
          "Did you forget to add '.def(torch::init<...>)' to its registration?",
          instance.type()->repr_str()));
  Method init_method(instance._ivalue(), init_fn);
  // NOLINTNEXTLINE(performance-move-const-arg)
  invokeScriptMethodFromPython(init_method, std::move(args), std::move(kwargs));
  return py::cast(instance);
}

/// Variant of StrongFunctionPtr, but for static methods of custom classes.
/// They do not belong to compilation units (the custom class method registry
/// serves that purpose in this case), so StrongFunctionPtr cannot be used here.
/// While it is usually unsafe to carry a raw pointer like this, the custom
/// class method registry that owns the pointer is never destroyed.
struct ScriptClassFunctionPtr {
  ScriptClassFunctionPtr(Function* function) : function_(function) {
    TORCH_INTERNAL_ASSERT(function_);
  }
  Function* function_;
};

static std::unordered_map<std::size_t, py::object>& customAbstractObjectCache() {
  static std::unordered_map<std::size_t, py::object> cache;
  return cache;
}

void initPythonCustomClassBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::class_<ScriptClassFunctionPtr>(
      m, "ScriptClassFunction", py::dynamic_attr())
      .def("__call__", [](py::args args, const py::kwargs& kwargs) {
        auto strongPtr = py::cast<ScriptClassFunctionPtr>(args[0]);
        Function& callee = *strongPtr.function_;
        py::object result = invokeScriptFunctionFromPython(
            callee, tuple_slice(std::move(args), 1), kwargs);
        return result;
      });

  py::class_<ScriptClass>(m, "ScriptClass")
      .def("__call__", &ScriptClass::__call__)
      .def(
          "__getattr__",
          [](ScriptClass& self, const std::string& name) {
            // Define __getattr__ so that static functions of custom classes can
            // be used in regular Python.
            auto type = self.class_type_.type_->castRaw<ClassType>();
            TORCH_INTERNAL_ASSERT(type);
            auto* fn = type->findStaticMethod(name);
            if (fn) {
              return ScriptClassFunctionPtr(fn);
            }

            throw AttributeError("%s does not exist", name.c_str());
          })
      .def_property_readonly("__doc__", [](const ScriptClass& self) {
        return self.class_type_.type_->expectRef<ClassType>().doc_string();
      });

  // This function returns a ScriptClass that wraps the constructor
  // of the given class, specified by the qualified name passed in.
  //
  // This is to emulate the behavior in python where instantiation
  // of a class is a call to a code object for the class, where that
  // code object in turn calls __init__. Rather than calling __init__
  // directly, we need a wrapper that at least returns the instance
  // rather than the None return value from __init__
  m.def(
      "_get_custom_class_python_wrapper",
      [](const std::string& ns, const std::string& qualname) {
        std::string full_qualname =
            "__torch__.torch.classes." + ns + "." + qualname;
        auto named_type = getCustomClass(full_qualname);
        TORCH_CHECK(
            named_type,
            fmt::format(
                "Tried to instantiate class '{}.{}', but it does not exist! "
                "Ensure that it is registered via torch::class_",
                ns,
                qualname));
        c10::ClassTypePtr class_type = named_type->cast<ClassType>();
        return ScriptClass(c10::StrongTypePtr(
            std::shared_ptr<CompilationUnit>(), std::move(class_type)));
      });

  m.def(
    "_delete_fake",
  [](py::object mirror_obj)  {
    // deregisterCustomClassMethods();
  });
  m.def(
      "_mirror_script_obj_with_python",
      [](py::handle real_obj, py::object mirror_obj) {
        // create a custom_type based on this object
        auto cprint = [](auto str) {
          std::cout << str << std::endl;
        };

        auto real_qual_class_name = py::cast<std::string>(real_obj.attr("__qualname__"));
        auto mirror_ns = "__torch__.torch.classes._mirrored_class.";
        auto qual_class_name = mirror_ns + real_qual_class_name + "_python_mirror";

        auto class_type_ptr = torch::getCustomClass(qual_class_name);
        if (!class_type_ptr) {
          cprint("registering qual_class_name:" + qual_class_name);

          class_type_ptr = at::ClassTypePtr(at::ClassType::create(
            c10::QualifiedName(qual_class_name),
            std::weak_ptr<CompilationUnit>(),
            /*is_module=*/false,
            std::move(qual_class_name + " is a mirrored implementation of " + real_qual_class_name
              + ". All methods on mirrored object is forwarded to python.")));

          auto register_py_fake_func = [cprint, qual_class_name](auto py_real_obj, auto class_type_ptr, auto py_fake_obj, auto method_name){
            auto schema = py::cast<c10::FunctionSchema>(py_real_obj.attr(py::cast(method_name)).attr("schema"));
            auto forward_py_func = [cprint, method_name, schema](jit::Stack& stack) {
              cprint("forwarding_function: " + method_name);
              py::gil_scoped_acquire g;

              auto self_arg = stack[0];
              auto py_args = createPyObjectForStack(std::move(stack));
              cprint("forwarding_function: 0");
              py::list arg_list = py::isinstance<py::tuple>(py_args) ? py_args : py::make_tuple(py_args);

              cprint("forwarding_function: 1");
              py::slice slice1(1, len(arg_list), 1);
              py::slice slice0(0, 1, 1);
              cprint("forwarding_function: 2");
              auto arg_list1 = py::cast<py::tuple>(arg_list[slice1]);
              auto arg_list2 = py::cast<py::tuple>(arg_list[slice0]);


              std::cout << self_arg << std::endl;
              auto fake_obj = customAbstractObjectCache()[std::hash<c10::ivalue::Object*>{}(self_arg.toObject().get())];
              TORCH_CHECK(!fake_obj.is_none(), "fake object is None");

              py::print(fake_obj);
              py::print(py::hasattr(fake_obj, py::cast(method_name)));
              auto ret = fake_obj.attr(py::cast(method_name))(*arg_list1);

              cprint("forwarding_function: 3");
              stack = jit::Stack();
              if (schema.returns().size() == 1) {
                auto ret_schema = schema.returns()[0];
                if (ret_schema.type() == c10::NoneType::get()) {
                  stack.emplace_back();
                } else {
                  auto ret_type = ret_schema.type();
                  stack.emplace_back(toIValue(ret, ret_type));
                }
              } else {
                auto ret_schema = schema.returns();
                for (size_t i = 0; i < ret_schema.size(); ++i) {
                  auto ret_type = ret_schema[i].type();
                  stack.emplace_back(toIValue(ret[py::cast(i)], ret_type));
                }
              }
              cprint("forwarding_function: 4");
            };

            auto method = std::make_unique<jit::BuiltinOpFunction>(
                c10::QualifiedName(qual_class_name+ "." + method_name),
                schema,
                std::move(forward_py_func),
                std::move(std::string("heihei.haha.") + method_name));
            class_type_ptr->addMethod(method.get());
            registerCustomClassMethod(std::move(method));
          };

          auto real_obj_cpp = py::cast<Object>(real_obj);
          for (auto func: real_obj_cpp.type()->methods()) {
            auto name = func->name();
            std::cout << name << std::endl;
            if (name != "__init__") {
              if (py::hasattr(mirror_obj, py::cast(name))){
                register_py_fake_func(real_obj, class_type_ptr, mirror_obj, name);
              } else {
                std::cout << "mirror object doesn't implement " << name << std::endl;
              }
            }
          }

          // TODO cache the registration and avoid capturing object into
          // the lambda.
          cprint(
            "registering custom class " + qual_class_name + " with torch::jit::registerCustomClass"
          );
          torch::registerCustomClass(class_type_ptr);
        }

        cprint(
          "Done registering custom class "
        );
        auto instance = Object(
          at::ivalue::Object::create(c10::StrongTypePtr(std::shared_ptr<CompilationUnit>(), class_type_ptr),
          /*numSlots=*/1)
        );
        auto obj = py::cast(instance);
        py::print("hash in python:");
        customAbstractObjectCache().insert(std::make_pair<std::size_t, py::object>(py::cast<std::size_t>(obj.attr("__hash__")()), std::move(mirror_obj)));
        // customAbstractObjectCache().insert({obj.attr("__hash__")(), mirror_obj});
        return obj;
      });
}

} // namespace torch::jit
