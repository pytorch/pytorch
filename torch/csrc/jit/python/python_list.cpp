#include <ATen/core/ivalue.h>
#include <pybind11/detail/common.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_list.h>
#include <stdexcept>

namespace torch {
namespace jit {

struct CustomMethodProxy;
struct CustomObjectProxy;

IValue ScriptListIterator::next() {
  if (iter_ == end_) {
    throw py::stop_iteration();
  }

  IValue result = *iter_;

  // Advance the iterator for next time.
  iter_++;

  return result;
}

void initScriptListBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::class_<ScriptListIterator>(m, "ScriptListIterator")
      .def(
          "__next__",
          [](ScriptListIterator& iter) {
            auto result = iter.next();
            return toPyObject(result);
          })
      .def("__iter__", [](ScriptListIterator& iter) { return iter; });

  py::class_<ScriptList, std::shared_ptr<ScriptList>>(m, "ScriptList")
      .def(py::init([](const TypePtr& type) {
        return std::make_shared<ScriptList>(type);
      }))
      .def(py::init([](py::list list, const TypePtr& type) {
        auto data = toIValue(std::move(list), type);
        return std::make_shared<ScriptList>(data);
      }))
      .def(
          "__repr__",
          [](const std::shared_ptr<ScriptList>& self) {
            return toPyObject(self->repr());
          })
      .def(
          "__bool__",
          [](const std::shared_ptr<ScriptList>& self) {
            return toPyObject(self->toBool());
          })
      .def(
          "__len__",
          [](const std::shared_ptr<ScriptList>& self) {
            return toPyObject(self->len());
          })
      .def(
          "__contains__",
          [](const std::shared_ptr<ScriptList>& self, py::object elem) {
            // TODO: What happens if elem isn't the right type?
            return toPyObject(self->contains(
                toIValue(std::move(elem), self->type()->getElementType())));
          })
      .def(
          "__getitem__",
          [](const std::shared_ptr<ScriptList>& self,
             ScriptList::size_type idx) {
            try {
              auto value = self->getItem(idx);
              return toPyObject(value);
            } catch (const std::out_of_range& e) {
              throw py::index_error();
            }
          },
          py::return_value_policy::
              reference_internal) // Return value is a reference to an object
                                  // that resides in the ScriptList
      .def(
          "__setitem__",
          [](const std::shared_ptr<ScriptList>& self,
             ScriptList::size_type idx,
             py::object value) {
            // TODO: What happens if value isn't the right type?
            try {
              self->setItem(
                  idx,
                  toIValue(std::move(value), self->type()->getElementType()));
            } catch (const std::out_of_range& e) {
              throw py::index_error();
            }
          })
      .def(
          "__delitem__",
          [](const std::shared_ptr<ScriptList>& self,
             ScriptList::size_type idx) { self->delItem(idx); })
      .def(
          "__iter__",
          [](const std::shared_ptr<ScriptList>& self) { return self->iter(); },
          py::keep_alive<0, 1>()) // ScriptList needs to be alive at least as
                                  // long as the iterator
      .def(
          "count",
          [](const std::shared_ptr<ScriptList>& self, py::object value) {
            // TODO: What happens if value isn't the right type?
            return self->count(
                toIValue(std::move(value), self->type()->getElementType()));
          })
      .def(
          "remove",
          [](const std::shared_ptr<ScriptList>& self, py::object value) {
            // TODO: What happens if value isn't the right type?
            return self->remove(
                toIValue(std::move(value), self->type()->getElementType()));
          })
      .def(
          "append",
          [](const std::shared_ptr<ScriptList>& self, py::object value) {
            // TODO: What happens if value isn't the right type?
            return self->append(
                toIValue(std::move(value), self->type()->getElementType()));
          })
      .def(
          "clear",
          [](const std::shared_ptr<ScriptList>& self) { self->clear(); })
      .def(
          "extend",
          [](const std::shared_ptr<ScriptList>& self, py::object obj) {
            // TODO: Handle dict, custom iterable, object of wrong type.
            py::list list = obj.cast<py::list>();
            self->extend(toIValue(list, self->type()));
          })
      .def(
          "pop",
          [](const std::shared_ptr<ScriptList>& self) {
            return toPyObject(self->pop());
          })
      .def(
          "pop",
          [](const std::shared_ptr<ScriptList>& self,
             ScriptList::size_type idx) { return toPyObject(self->pop(idx)); })
      .def(
          "insert",
          [](const std::shared_ptr<ScriptList>& self,
             py::object obj,
             ScriptList::size_type idx) {
            // TODO: What happens if obj is the wrong type?
            self->insert(
                toIValue(std::move(obj), self->type()->getElementType()), idx);
          });
}

} // namespace jit
} // namespace torch
