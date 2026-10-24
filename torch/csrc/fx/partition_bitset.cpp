#include <torch/csrc/fx/partition_bitset.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace torch::fx {
namespace {

struct PartitionBitset {
  PyObject_HEAD
  Py_ssize_t size;
  uint64_t* words;

  size_t wordCount() const {
    return size / 64 + (size % 64 != 0);
  }

  void maskTail() {
    if (size % 64 != 0) {
      words[wordCount() - 1] &= (uint64_t{1} << (size % 64)) - 1;
    }
  }
};

struct PartitionBitsetIterator {
  PyObject_HEAD
  PartitionBitset* bits;
  size_t word_index;
  uint64_t remaining;
};

extern PyTypeObject PartitionBitsetType;
extern PyTypeObject PartitionBitsetIteratorType;

// The caller initializes every word before exposing the result to Python.
PartitionBitset* allocateBitset(Py_ssize_t size) {
  auto* result = PyObject_New(PartitionBitset, &PartitionBitsetType);
  if (result == nullptr) {
    return nullptr;
  }
  result->size = size;
  result->words = nullptr;
  if (size != 0) {
    result->words = static_cast<uint64_t*>(
        PyMem_Malloc(result->wordCount() * sizeof(uint64_t)));
    if (result->words == nullptr) {
      Py_DECREF(result);
      PyErr_NoMemory();
      return nullptr;
    }
  }
  return result;
}

void bitsetDealloc(PyObject* object) {
  PyMem_Free(reinterpret_cast<PartitionBitset*>(object)->words);
  Py_TYPE(object)->tp_free(object);
}

PyObject* bitsetNew(PyTypeObject*, PyObject* args, PyObject* kwargs) {
  Py_ssize_t size;
  int fill = 0;
  static const char* keywords[] = {"size", "fill", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "n|p:_PartitionBitset",
          const_cast<char**>(keywords),
          &size,
          &fill)) {
    return nullptr;
  }
  if (size < 0) {
    PyErr_SetString(PyExc_ValueError, "Bitset size must be nonnegative");
    return nullptr;
  }
  auto* result = allocateBitset(size);
  if (result == nullptr) {
    return nullptr;
  }
  if (size != 0) {
    std::fill_n(result->words, result->wordCount(), fill ? ~uint64_t{0} : 0);
    result->maskTail();
  }
  return reinterpret_cast<PyObject*>(result);
}

Py_ssize_t bitsetLength(PyObject* object) {
  return reinterpret_cast<PartitionBitset*>(object)->size;
}

Py_ssize_t bitsetIndex(PartitionBitset* self, PyObject* index) {
  auto value = PyNumber_AsSsize_t(index, PyExc_IndexError);
  if (value == -1 && PyErr_Occurred()) {
    return -1;
  }
  if (value < 0 || value >= self->size) {
    PyErr_SetString(PyExc_IndexError, "Bitset index out of range");
    return -1;
  }
  return value;
}

PyObject* bitsetGet(PyObject* object, PyObject* index) {
  auto* self = reinterpret_cast<PartitionBitset*>(object);
  auto i = bitsetIndex(self, index);
  if (i < 0) {
    return nullptr;
  }
  return PyBool_FromLong(
      (self->words[i / 64] & (uint64_t{1} << (i % 64))) != 0);
}

int bitsetSet(PyObject* object, PyObject* index, PyObject* value) {
  if (value == nullptr) {
    PyErr_SetString(PyExc_TypeError, "Bitset bits cannot be deleted");
    return -1;
  }
  auto* self = reinterpret_cast<PartitionBitset*>(object);
  auto i = bitsetIndex(self, index);
  if (i < 0) {
    return -1;
  }
  int fill = PyObject_IsTrue(value);
  if (fill < 0) {
    return -1;
  }
  auto bit = uint64_t{1} << (i % 64);
  if (fill) {
    self->words[i / 64] |= bit;
  } else {
    self->words[i / 64] &= ~bit;
  }
  return 0;
}

bool checkSize(const PartitionBitset* a, const PartitionBitset* b) {
  if (a->size != b->size) {
    PyErr_SetString(PyExc_ValueError, "Bitsets must have the same size");
    return false;
  }
  return true;
}

template <char operation, bool inplace = false>
PyObject* bitsetBinary(PyObject* left, PyObject* right) {
  if (!Py_IS_TYPE(left, &PartitionBitsetType) ||
      !Py_IS_TYPE(right, &PartitionBitsetType)) {
    if constexpr (operation == '-') {
      PyErr_SetString(PyExc_TypeError, "Expected a _PartitionBitset");
      return nullptr;
    } else {
      Py_RETURN_NOTIMPLEMENTED;
    }
  }
  auto* a = reinterpret_cast<PartitionBitset*>(left);
  auto* b = reinterpret_cast<PartitionBitset*>(right);
  if (!checkSize(a, b)) {
    return nullptr;
  }
  auto* result = inplace ? a : allocateBitset(a->size);
  if (result == nullptr) {
    return nullptr;
  }
  const auto count = a->wordCount();
  for (size_t i = 0; i < count; ++i) {
    if constexpr (operation == '&') {
      result->words[i] = a->words[i] & b->words[i];
    } else if constexpr (operation == '|') {
      result->words[i] = a->words[i] | b->words[i];
    } else {
      result->words[i] = a->words[i] & ~b->words[i];
    }
  }
  if constexpr (inplace) {
    Py_INCREF(result);
  }
  return reinterpret_cast<PyObject*>(result);
}

PyObject* bitsetCopy(PyObject* object, PyObject*) {
  auto* self = reinterpret_cast<PartitionBitset*>(object);
  auto* result = allocateBitset(self->size);
  if (result != nullptr && self->size != 0) {
    std::memcpy(
        result->words, self->words, self->wordCount() * sizeof(uint64_t));
  }
  return reinterpret_cast<PyObject*>(result);
}

PyObject* bitsetInvert(PyObject* object) {
  auto* self = reinterpret_cast<PartitionBitset*>(object);
  auto* result = allocateBitset(self->size);
  if (result == nullptr) {
    return nullptr;
  }
  const auto count = self->wordCount();
  for (size_t i = 0; i < count; ++i) {
    result->words[i] = ~self->words[i];
  }
  result->maskTail();
  return reinterpret_cast<PyObject*>(result);
}

PyObject* bitsetCompare(PyObject* left, PyObject* right, int operation) {
  if (!Py_IS_TYPE(right, &PartitionBitsetType) ||
      (operation != Py_EQ && operation != Py_NE)) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  auto* a = reinterpret_cast<PartitionBitset*>(left);
  auto* b = reinterpret_cast<PartitionBitset*>(right);
  bool equal = a->size == b->size &&
      (a->size == 0 ||
       std::memcmp(a->words, b->words, a->wordCount() * sizeof(uint64_t)) == 0);
  return PyBool_FromLong(operation == Py_EQ ? equal : !equal);
}

PyObject* bitsetIntersects(
    PyObject* object,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* keywords) {
  if (nargs > 2) {
    PyErr_SetString(PyExc_TypeError, "intersects() takes at most 2 arguments");
    return nullptr;
  }
  PyObject* other = nargs > 0 ? args[0] : nullptr;
  PyObject* mask = nargs > 1 ? args[1] : nullptr;
  if (keywords != nullptr) {
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(keywords); ++i) {
      auto* name = PyTuple_GET_ITEM(keywords, i);
      PyObject** target;
      if (PyUnicode_CompareWithASCIIString(name, "other") == 0) {
        target = &other;
      } else if (PyUnicode_CompareWithASCIIString(name, "mask") == 0) {
        target = &mask;
      } else {
        PyErr_Format(PyExc_TypeError, "Unexpected keyword argument: %U", name);
        return nullptr;
      }
      if (*target != nullptr) {
        PyErr_Format(PyExc_TypeError, "Multiple values for argument: %U", name);
        return nullptr;
      }
      *target = args[nargs + i];
    }
  }
  if (other == nullptr || !Py_IS_TYPE(other, &PartitionBitsetType) ||
      (mask != nullptr && mask != Py_None &&
       !Py_IS_TYPE(mask, &PartitionBitsetType))) {
    PyErr_SetString(PyExc_TypeError, "Expected _PartitionBitset operands");
    return nullptr;
  }
  auto* a = reinterpret_cast<PartitionBitset*>(object);
  auto* b = reinterpret_cast<PartitionBitset*>(other);
  auto* active =
      mask == Py_None ? nullptr : reinterpret_cast<PartitionBitset*>(mask);
  if (!checkSize(a, b) || (active != nullptr && !checkSize(a, active))) {
    return nullptr;
  }
  for (size_t i = 0; i < a->wordCount(); ++i) {
    auto common = a->words[i] & b->words[i];
    if ((active == nullptr ? common : common & active->words[i]) != 0) {
      Py_RETURN_TRUE;
    }
  }
  Py_RETURN_FALSE;
}

PyObject* bitsetSetBits(PyObject* object, PyObject*) {
  auto* iterator =
      PyObject_New(PartitionBitsetIterator, &PartitionBitsetIteratorType);
  if (iterator == nullptr) {
    return nullptr;
  }
  // Retain the owner when iterating over a temporary bitset expression.
  iterator->bits = reinterpret_cast<PartitionBitset*>(Py_NewRef(object));
  iterator->word_index = 0;
  iterator->remaining =
      iterator->bits->size == 0 ? 0 : iterator->bits->words[0];
  return reinterpret_cast<PyObject*>(iterator);
}

void iteratorDealloc(PyObject* object) {
  Py_DECREF(reinterpret_cast<PartitionBitsetIterator*>(object)->bits);
  Py_TYPE(object)->tp_free(object);
}

PyObject* iteratorNext(PyObject* object) {
  auto* self = reinterpret_cast<PartitionBitsetIterator*>(object);
  while (self->word_index < self->bits->wordCount()) {
    if (self->remaining != 0) {
      auto index = self->word_index * 64 + std::countr_zero(self->remaining);
      self->remaining &= self->remaining - 1;
      return PyLong_FromSize_t(index);
    }
    ++self->word_index;
    if (self->word_index < self->bits->wordCount()) {
      self->remaining = self->bits->words[self->word_index];
    }
  }
  // A null result without an error signals normal iterator exhaustion.
  return nullptr;
}

PyMethodDef bitsetMethods[] = {
    {"copy", bitsetCopy, METH_NOARGS, nullptr},
    {"difference", bitsetBinary<'-'>, METH_O, nullptr},
    {"intersects",
     reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(bitsetIntersects)),
     METH_FASTCALL | METH_KEYWORDS,
     nullptr},
    {"set_bits", bitsetSetBits, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyNumberMethods bitsetNumbers = {};
PyMappingMethods bitsetMapping = {bitsetLength, bitsetGet, bitsetSet};
PyTypeObject PartitionBitsetType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._PartitionBitset", /* tp_name */
    sizeof(PartitionBitset), /* tp_basicsize */
    0, /* tp_itemsize */
    bitsetDealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    &bitsetNumbers, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    &bitsetMapping, /* tp_as_mapping */
    PyObject_HashNotImplemented, /* tp_hash */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    bitsetCompare, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    bitsetMethods, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    bitsetNew, /* tp_new */
};

PyTypeObject PartitionBitsetIteratorType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._PartitionBitsetIterator", /* tp_name */
    sizeof(PartitionBitsetIterator), /* tp_basicsize */
    0, /* tp_itemsize */
    iteratorDealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_DISALLOW_INSTANTIATION, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    PyObject_SelfIter, /* tp_iter */
    iteratorNext, /* tp_iternext */
};

} // namespace

bool initPartitionBitsetBindings(PyObject* module) {
  bitsetNumbers.nb_and = bitsetBinary<'&'>;
  bitsetNumbers.nb_or = bitsetBinary<'|'>;
  bitsetNumbers.nb_inplace_or = bitsetBinary<'|', true>;
  bitsetNumbers.nb_invert = bitsetInvert;
  return PyModule_AddType(module, &PartitionBitsetIteratorType) == 0 &&
      PyModule_AddType(module, &PartitionBitsetType) == 0;
}

} // namespace torch::fx
