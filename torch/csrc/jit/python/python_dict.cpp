#include <ATen/core/ivalue.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_dict.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/utils/pybind.h>
#include <sstream>
#include <stdexcept>

namespace torch::jit {

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
      .def(py::init([](py::dict dict) {
        TypePtr type = nullptr;

        if (!dict.empty()) {
          // If the source dictionary is nonempty, try to infer its type.
          auto inferred_type = tryToInferType(dict);

          if (!inferred_type.success()) {
            std::stringstream ss;
            ss << "Unable to infer type of dictionary: "
               << inferred_type.reason();
            throw JITException(ss.str());
          }

          type = inferred_type.type();
        } else {
          // If is empty, assume the type is Dict[str, Tensor] as is done in
          // TorchScript code.
          type = DictType::create(StringType::get(), TensorType::getInferred());
        }

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
            try {
              return toPyObject(self->contains(
                  toIValue(std::move(key), self->type()->getKeyType())));
            } catch (const py::cast_error& e) {
              throw py::key_error();
            }
          })
      .def(
          "__getitem__",
          [](const std::shared_ptr<ScriptDict>& self, py::object key) {
            IValue value;

            // Convert key to IValue.
            try {
              value = toIValue(std::move(key), self->type()->getKeyType());
            } catch (const py::cast_error& e) {
              // It would be nice to throw py::type_error here but py::key_error
              // needs to be thrown for parity with eager mode.
              throw py::key_error();
            }

            // Call getItem on self.
            try {
              value = self->getItem(value);
            } catch (const std::out_of_range& e) { // Key doesn't exist.
              throw py::key_error();
            }

            return toPyObject(std::move(value));
          },
          py::return_value_policy::
              reference_internal) // Return value is a reference to an object
                                  // that resides in the ScriptDict
      .def(
          "__setitem__",
          [](const std::shared_ptr<ScriptDict>& self,
             py::object key,
             py::object value) {
            IValue key_ivalue, value_ivalue;

            // Try to convert the key to an IValue.
            try {
              key_ivalue = toIValue(std::move(key), self->type()->getKeyType());
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }

            // Try to convert the value to an IValue.
            try {
              value_ivalue =
                  toIValue(std::move(value), self->type()->getValueType());
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }

            self->setItem(key_ivalue, value_ivalue);
          })
      .def(
          "__delitem__",
          [](const std::shared_ptr<ScriptDict>& self, py::object key) {
            IValue key_ivalue;

            // Try to convert the key to an IValue.
            try {
              key_ivalue = toIValue(std::move(key), self->type()->getKeyType());
            } catch (const py::cast_error& e) {
              throw py::type_error();
            }

            // If removed = false, that means the key didn't exist in the
            // dictionary.
            bool removed = self->delItem(key_ivalue);

            if (!removed) {
              throw py::key_error();
            }
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

} // namespace torch::jit
