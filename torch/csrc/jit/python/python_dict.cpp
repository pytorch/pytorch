#include <pybind11/detail/common.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_dict.h>
#include <stdexcept>
#include <ATen/core/ivalue.h>

namespace torch {
namespace jit {

struct CustomMethodProxy;
struct CustomObjectProxy;

IValue ScriptDictIterator::next() {
  if (iter_ == end_) {
    throw py::stop_iteration();
  }

  // Since this is the iterator for .items(), the current key and value
  // should be returned as a tuple.
  IValue result = c10::ivalue::Tuple::create({iter_->key(), iter_->value()});

  // Advance the iterator for next time.
  iter_++;

  return result;
}

IValue ScriptDictKeyIterator::next() {
  if (iter_ == end_) {
    throw py::stop_iteration();
  }

  // Since this is the iterator for .keys() and __iter__(), return only the key.
  IValue result = iter_->key();

  // Advance the iterator for next time.
  iter_++;

  return result;
}

void initScriptDictBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::class_<ScriptDictKeyIterator>(m, "ScriptDictKeyIterator")
      .def(
          "__next__",
          [](ScriptDictKeyIterator& iter) {
            auto result = iter.next();
            return toPyObject(result);
          })
      .def("__iter__", [](ScriptDictKeyIterator& iter) { return iter; });

  py::class_<ScriptDictIterator>(m, "ScriptDictIterator")
      .def(
          "__next__",
          [](ScriptDictIterator& iter) {
            auto result = iter.next();
            return toPyObject(result);
          })
      .def("__iter__", [](ScriptDictIterator& iter) { return iter; });

  py::class_<ScriptDict, std::shared_ptr<ScriptDict>>(m, "ScriptDict")
      .def(py::init([](const TypePtr& type) {
        return std::make_shared<ScriptDict>(type);
      }))
      .def(py::init([](py::dict dict, const TypePtr& type) {
        auto data = toIValue(std::move(dict), type);
        return std::make_shared<ScriptDict>(data);
      }))
      .def(
          "__repr__",
          [](const std::shared_ptr<ScriptDict>& self) {
            return toPyObject(self->repr());
          })
      .def(
          "__bool__",
          [](const std::shared_ptr<ScriptDict>& self) {
            return toPyObject(self->toBool());
          })
      .def(
          "__len__",
          [](const std::shared_ptr<ScriptDict>& self) {
            return toPyObject(self->len());
          })
      .def(
          "__contains__",
          [](const std::shared_ptr<ScriptDict>& self, py::object key) {
            // TODO: What happens if key isn't the right type?
            return toPyObject(self->contains(
                toIValue(std::move(key), self->type()->getKeyType())));
          })
      .def(
          "__getitem__",
          [](const std::shared_ptr<ScriptDict>& self, py::object key) {
            // TODO: What happens if key isn't the right type?
            try {
              auto value = self->getItem(
                  toIValue(std::move(key), self->type()->getKeyType()));
              return toPyObject(value);
            } catch (const std::out_of_range& e) {
              throw py::key_error();
            }
          },
          py::return_value_policy::
              reference_internal) // Return value is a reference to an object
                                  // that resides in the ScriptDict
      .def(
          "__delitem__",
          [](const std::shared_ptr<ScriptDict>& self, py::object key) {
            // TODO: What happens if key isn't the right type?
            self->delItem(toIValue(std::move(key), self->type()->getKeyType()));
          })
      .def(
          "__iter__",
          [](const std::shared_ptr<ScriptDict>& self) { return self->iter(); },
          py::keep_alive<0, 1>()) // ScriptDict needs to be alive at least as
                                  // long as the iterator
      .def(
          "items",
          [](const std::shared_ptr<ScriptDict>& self) { return self->items(); },
          py::keep_alive<0, 1>()) // ScriptDict needs to be alive at least as
                                  // long as the iterator
      .def(
          "keys",
          [](const std::shared_ptr<ScriptDict>& self) { return self->iter(); },
          py::keep_alive<0, 1>()); // ScriptDict needs to be alive at least as
                                   // long as the iterator
}

} // namespace jit
} // namespace torch
