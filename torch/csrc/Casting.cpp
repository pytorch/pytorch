#include "torch/csrc/Casting.h"

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/pybind.h"

#include <c10/core/Casting.h>
#include <c10/util/Exception.h>

#include <cstring>
#include <limits>
#include <structmember.h>
#include <sstream>

PyObject *THPCasting_New(c10::Casting casting)
{
  auto type = (PyTypeObject*)&THPCastingType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPCasting*>(self.get());
  self_->casting = casting;
  return self.release();
}

PyObject *THPCasting_repr(THPCasting *self)
{
  std::ostringstream oss;
  oss << "casting(" << self->casting << ")";
  return THPUtils_packString(oss.str().c_str());
}

PyObject *THPCasting_str(THPCasting *self)
{
  std::ostringstream oss;
  oss << self->casting;
  return THPUtils_packString(oss.str().c_str());
}

PyObject *THPCasting_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "Casting(Casting casting)",
    "Casting(std::string casting)"
  });
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0 || r.idx == 1) {
    return THPCasting_New(r.casting(0));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static Py_ssize_t THPCasting_hash(THPCasting *self)
{
  HANDLE_TH_ERRORS
  return static_cast<Py_ssize_t>(
      std::hash<std::size_t>{}(static_cast<std::size_t>(self->casting)) %
      std::numeric_limits<Py_ssize_t>::max());
  END_HANDLE_TH_ERRORS_RET(-1)
}

typedef PyObject *(*getter)(PyObject *, void *);

PyTypeObject THPCastingType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch.casting",                       /* tp_name */
  sizeof(THPCasting),                    /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  (reprfunc)THPCasting_repr,             /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  (hashfunc)THPCasting_hash,             /* tp_hash  */
  0,                                     /* tp_call */
  (reprfunc)THPCasting_str,              /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                    /* tp_flags */
  nullptr,                               /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  0,                                     /* tp_methods */
  0,                                     /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPCasting_pynew,                      /* tp_new */
};

void THPCasting_init(PyObject *module)
{
  if (PyType_Ready(&THPCastingType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPCastingType);
  if (PyModule_AddObject(module, "casting", (PyObject *)&THPCastingType) != 0) {
    throw python_error();
  }
}
