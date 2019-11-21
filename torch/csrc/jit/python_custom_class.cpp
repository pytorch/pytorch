#include <torch/csrc/jit/python_custom_class.h>
#include <torch/csrc/jit/script/sugared_value.h>

namespace torch {
namespace jit {

struct CustomMethodProxy;
struct CustomObjectProxy;

void initPythonCustomClassBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  m.def("_get_custom_class_python_wrapper", [](const std::string& str) {
    auto cu = classCU();
    c10::NamedTypePtr named_type =
        cu->get_type("__torch__.torch.classes." + str);
    if (!named_type || !named_type->cast<ClassType>()) {
      std::stringstream err;
      err << "Class " << str << " not registered!";
      throw std::runtime_error(err.str());
    }
    c10::ClassTypePtr class_type = named_type->cast<ClassType>();
    Function* ctor_method = class_type->getMethod("__init__");
    if (!ctor_method) {
      std::stringstream err;
      err << "Class ";
      if (auto name = class_type->name()) {
        err << name->qualifiedName() << " ";
      }
      err << "does not have an __init__ method defined!";
      throw std::runtime_error(err.str());
    }

    // Need to wrap __init__ in another function that actually returns the
    // object so that torch.classes.Foo() doesn't just return None
    auto wrapper_fn_name =
        class_type->name()->qualifiedName() + ".__init__wrapper";
    Function* ctor_wrapper;
    if (classCU()->find_function(wrapper_fn_name)) {
      ctor_wrapper = &classCU()->get_function(wrapper_fn_name);
    } else {
      auto graph = std::make_shared<Graph>();
      ctor_wrapper = classCU()->create_function(wrapper_fn_name, graph);
      auto orig_graph = ctor_method->graph();
      for (size_t i = 0; i < orig_graph->inputs().size(); ++i) {
        if (i == 0) {
          continue;
        }
        Value* orig_inp = orig_graph->inputs()[i];
        graph->addInput()->copyMetadata(orig_inp);
      }
      Value* self =
          graph->insertNode(graph->createObject(class_type))->output();
      std::vector<NamedValue> named_values;
      for (Value* inp : graph->inputs()) {
        named_values.emplace_back(inp->node()->sourceRange(), inp);
      }
      script::MethodValue(self, "__init__")
          .call(SourceRange(), *ctor_wrapper, named_values, {}, 0);
      for (size_t i = 0; i < graph->outputs().size(); ++i) {
        graph->eraseOutput(graph->outputs().size() - i - 1);
      }
      graph->registerOutput(self);
    }

    return StrongFunctionPtr(classCU(), ctor_wrapper);
  });
}

} // namespace jit
} // namespace torch
