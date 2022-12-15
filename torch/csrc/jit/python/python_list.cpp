#include <ATen/core/ivalue.h>
#include <c10/util/irange.h>
#include <pybind11/detail/common.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_list.h>
#include <torch/csrc/utils/pybind.h>
#include <stdexcept>

namespace torch {
namespace jit {

IValue ScriptListIterator::next() {
  if (iter_ == end_) {
    throw py::stop_iteration();
  }

  IValue result = *iter_;

  // Advance the iterator for next time.
  iter_++;

  return result;
}

bool ScriptListIterator::done() const {
  return iter_ == end_;
}

namespace {
py::list scriptListToPyList(const ScriptList& src) {
  py::list out(src.len());
  auto iter = src.iter();

  size_t i = 0;
  while (!iter.done()) {
    auto val = iter.next();
    // TODO: Handle nested dictionaries.
    if (val.isList()) {
      out[i] = scriptListToPyList(val);
    } else {
      out[i] = toPyObject(val);
    }
    ++i;
  }

  return out;
}
} // namespace

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
      .def(py::init([](py::list list) {
        TypePtr type = nullptr;

        if (list.size() > 0) {
          // If the source list is nonempty, try to infer its type.
          auto inferred_type = tryToInferType(list);

          if (!inferred_type.success()) {
            std::stringstream ss;
            ss << "Unable to infer type of list: " << inferred_type.reason();
            throw JITException(ss.str());
          }

          type = inferred_type.type();
        } else {
          // If is empty, assume the type is List[Tensor] as is done in
          // TorchScript code.
          type = ListType::create(TensorType::getInferred());
        }

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
            try {
              return toPyObject(self->contains(
                  toIValue(std::move(elem), self->type()->getElementType())));
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }
          })
      .def(
          "__getitem__",
          [](const std::shared_ptr<ScriptList>& self,
             ScriptList::diff_type idx) {
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
          "__getitem__",
          [](const std::shared_ptr<ScriptList>& self, const py::slice& slice) {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;

            if (!slice.compute(
                    self->len(), &start, &stop, &step, &slicelength)) {
              throw py::error_already_set();
            }

            auto seq = std::make_shared<ScriptList>(self->type());

            for (const auto i : c10::irange(slicelength)) {
              (void)i; // Suppress unused variable warning
              seq->append(self->getItem(start));
              start += step;
            }

            return seq;
          })
      .def(
          "__setitem__",
          [](const std::shared_ptr<ScriptList>& self,
             ScriptList::diff_type idx,
             py::object value) {
            try {
              self->setItem(
                  idx,
                  toIValue(std::move(value), self->type()->getElementType()));
            } catch (const std::out_of_range& e) {
              throw py::index_error();
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }
          })
      .def(
          "__setitem__",
          [](const std::shared_ptr<ScriptList>& self,
             const py::slice& slice,
             const py::list& value) {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;

            if (!slice.compute(
                    self->len(), &start, &stop, &step, &slicelength)) {
              throw py::error_already_set();
            }

            if (slicelength != value.size()) {
              throw std::runtime_error(
                  "Left and right hand size of slice assignment have different sizes");
            }

            for (const auto i : c10::irange(slicelength)) {
              try {
                self->setItem(
                    start, toIValue(value[i], self->type()->getElementType()));
              } catch (const py::cast_error& e) {
                throw py::type_error();
              }
              start += step;
            }
          })
      .def(
          "__delitem__",
          [](const std::shared_ptr<ScriptList>& self,
             ScriptList::diff_type idx) {
            try {
              self->delItem(idx);
            } catch (const std::out_of_range& e) {
              throw py::index_error();
            }
          })
      .def(
          "__iter__",
          [](const std::shared_ptr<ScriptList>& self) { return self->iter(); },
          py::keep_alive<0, 1>()) // ScriptList needs to be alive at least as
                                  // long as the iterator
      .def(
          "count",
          [](const std::shared_ptr<ScriptList>& self, py::object value) {
            try {
              return self->count(
                  toIValue(std::move(value), self->type()->getElementType()));

            } catch (const py::cast_error& e) {
              throw py::type_error();
            }
          })
      .def(
          "remove",
          [](const std::shared_ptr<ScriptList>& self, py::object value) {
            try {
              return self->remove(
                  toIValue(std::move(value), self->type()->getElementType()));
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }
          })
      .def(
          "append",
          [](const std::shared_ptr<ScriptList>& self, py::object value) {
            try {
              return self->append(
                  toIValue(std::move(value), self->type()->getElementType()));
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }
          })
      .def(
          "clear",
          [](const std::shared_ptr<ScriptList>& self) { self->clear(); })
      .def(
          "extend",
          [](const std::shared_ptr<ScriptList>& self, py::list list) {
            try {
              self->extend(toIValue(std::move(list), self->type()));
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }
          })
      .def(
          "extend",
          [](const std::shared_ptr<ScriptList>& self,
             const py::iterable& iter) {
            ScriptList iter_list(self->type());

            try {
              for (py::handle obj : iter) {
                iter_list.append(toIValue(
                    py::reinterpret_borrow<py::object>(obj),
                    self->type()->getElementType()));
              }
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }

            self->extend(toIValue(py::cast(iter_list), self->type()));
          })
      .def(
          "pop",
          [](const std::shared_ptr<ScriptList>& self) {
            return toPyObject(self->pop());
          })
      .def(
          "pop",
          [](const std::shared_ptr<ScriptList>& self,
             ScriptList::diff_type idx) { return toPyObject(self->pop(idx)); })
      .def(
          "insert",
          [](const std::shared_ptr<ScriptList>& self,
             ScriptList::diff_type idx,
             py::object obj) {
            try {
              self->insert(
                  toIValue(std::move(obj), self->type()->getElementType()),
                  idx);
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }
          })
      .def(py::pickle(
          [](const ScriptList& data) { // __getstate__
            return scriptListToPyList(data);
          },
          [](py::list list) { // __setstate__
            TypePtr type = nullptr;

            if (list.size() > 0) {
              // If the source list is nonempty, try to infer its type.
              auto inferred_type = tryToInferType(list);

              if (!inferred_type.success()) {
                std::stringstream ss;
                ss << "Unable to infer type of list: "
                   << inferred_type.reason();
                throw JITException(ss.str());
              }

              type = inferred_type.type();
            } else {
              // If is empty, assume the type is List[Tensor] as is done in
              // TorchScript code.
              type = ListType::create(TensorType::getInferred());
            }

            auto data = toIValue(std::move(list), type);
            return std::make_shared<ScriptList>(data);
          }));
}

} // namespace jit
} // namespace torch
