#include <torch/python/init.h>

#include <torch/nn/module.h>
#include <torch/ordered_dict.h>

#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>

#include <string>
#include <vector>

namespace py = pybind11;

namespace pybind11 {
namespace detail {
#define ITEM_TYPE_CASTER(T, Name)                                             \
  template <>                                                                 \
  struct type_caster<typename torch::OrderedDict<std::string, T>::Item> {     \
   public:                                                                    \
    using Item = typename torch::OrderedDict<std::string, T>::Item;           \
    using PairCaster = make_caster<std::pair<std::string, T>>;                \
    PYBIND11_TYPE_CASTER(Item, _("Ordered" #Name "DictItem"));                \
    bool load(handle src, bool convert) {                                     \
      return PairCaster().load(src, convert);                                 \
    }                                                                         \
    static handle cast(Item src, return_value_policy policy, handle parent) { \
      return PairCaster::cast(                                                \
          src.pair(), std::move(policy), std::move(parent));                  \
    }                                                                         \
  }

ITEM_TYPE_CASTER(torch::Tensor, Tensor);
ITEM_TYPE_CASTER(std::shared_ptr<torch::nn::Module>, Module);
} // namespace detail
} // namespace pybind11

namespace torch {
namespace python {
namespace {
Device py_object_to_device(py::object object) {
  PyObject* obj = object.ptr();
  if (THPDevice_Check(obj)) {
    return reinterpret_cast<THPDevice*>(obj)->device;
  }
  throw TypeError("Expected device");
}
Dtype py_object_to_dtype(py::object object) {
  PyObject* obj = object.ptr();
  if (THPDtype_Check(obj)) {
    return reinterpret_cast<THPDtype*>(obj)->scalar_type;
  }
  throw TypeError("Expected dtype");
}
} // namespace

template <typename T>
void bind_ordered_dict(
    py::module module,
    const char* dict_name,
    const char* item_name) {
  using ODict = OrderedDict<std::string, T>;
  // clang-format off
  py::class_<ODict>(module, dict_name)
      .def("items", &ODict::items)
      .def("keys", &ODict::keys)
      .def("values", &ODict::values)
      .def("__iter__", [](const ODict& dict) {
            return py::make_iterator(dict.begin(), dict.end());
          }, py::keep_alive<0, 1>())
      .def("__len__", &ODict::size)
      .def("__contains__", &ODict::contains)
      .def("__getitem__", [](const ODict& dict, const std::string& key) {
        return dict[key];
      })
      .def("__getitem__", [](const ODict& dict, size_t index) {
        return dict[index];
      });
  // clang-format on
}

void init_bindings(PyObject* module) {
  py::module m = py::handle(module).cast<py::module>();
  py::module cpp = m.def_submodule("cpp");

  bind_ordered_dict<Tensor>(cpp, "OrderedTensorDict", "OrderedTensorDictItem");
  bind_ordered_dict<std::shared_ptr<nn::Module>>(
      cpp, "OrderedModuleDict", "OrderedModuleDictItem");

  // clang-format off
  py::module nn = cpp.def_submodule("nn");
  py::class_<nn::Module, std::shared_ptr<nn::Module>>(nn, "Module")
      .def("train", &nn::Module::train, py::arg("mode") = true)
      .def("eval", &nn::Module::eval)
      .def("clone", &nn::Module::clone)
      .def_property_readonly("training", &nn::Module::is_training)
      .def("zero_grad", &nn::Module::zero_grad)
      .def_property_readonly("_parameters", [](nn::Module& module) {
          return module.named_parameters(/*recurse=*/false);
        })
      .def("parameters", &nn::Module::parameters, py::arg("recurse") = true)
      .def("named_parameters", &nn::Module::named_parameters, py::arg("recurse") = true)
      .def_property_readonly("_buffers", [](nn::Module& module) {
            return module.named_buffers(/*recurse=*/false);
          })
      .def("buffers", &nn::Module::buffers, py::arg("recurse") = true)
      .def("named_buffers", &nn::Module::named_buffers, py::arg("recurse") = true)
      .def("modules", &nn::Module::modules)
      .def("named_modules", [](nn::Module& module, py::object memo, std::string prefix) {
            return module.named_modules(std::move(prefix));
          },
          py::arg("memo") = py::none(),
          py::arg("prefix") = std::string())
      .def_property_readonly("_modules", &nn::Module::children)
      .def("children", &nn::Module::children)
      .def("named_children", &nn::Module::named_children)
      .def("to", [](nn::Module& module, py::object object, bool non_blocking) {
            if (THPDevice_Check(object.ptr())) {
              module.to(reinterpret_cast<THPDevice*>(object.ptr())->device, non_blocking);
            } else {
              module.to(py_object_to_dtype(object), non_blocking);
            }
          },
          py::arg("dtype_or_device"),
          py::arg("non_blocking") = false)
      .def("to", [](nn::Module& module,
                    py::object device,
                    py::object dtype,
                    bool non_blocking) {
            module.to(
                py_object_to_device(device),
                py_object_to_dtype(dtype),
                non_blocking);
          },
          py::arg("device"),
          py::arg("dtype"),
          py::arg("non_blocking") = false)
      .def("cuda", [](nn::Module& module) { module.to(kCUDA); })
      .def("cpu", [](nn::Module& module) { module.to(kCPU); })
      .def("float", [](nn::Module& module) { module.to(kFloat32); })
      .def("double", [](nn::Module& module) { module.to(kFloat64); })
      .def("half", [](nn::Module& module) { module.to(kFloat16); })
      .def("__str__", &nn::Module::name)
      .def("__repr__", &nn::Module::name);
  // clang-format on
}
} // namespace python
} // namespace torch
