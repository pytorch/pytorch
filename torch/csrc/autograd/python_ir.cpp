#include "torch/csrc/autograd/python_ir.h"
#include "torch/csrc/utils/python_strings.h"

#include <sstream>

using namespace torch::autograd;

PyObject* THPExprClass = nullptr;
PyObject* THPArgClass = nullptr;

PyObject* THPExpr_Wrap(const std::shared_ptr<Expr>& e)
{
  if (!e) {
    Py_RETURN_NONE;
  } else {
    auto type = (PyTypeObject*) THPExprClass;
    THPExpr* obj = (THPExpr*)type->tp_alloc(type, 0);
    if (obj) {
      obj->cdata = e;
    }
    return (PyObject*) obj;
  }
}

PyObject* THPArg_Wrap(const std::shared_ptr<Arg>& a)
{
  if (!a) {
    Py_RETURN_NONE;
  } else {
    auto type = (PyTypeObject*) THPArgClass;
    THPArg* obj = (THPArg*)type->tp_alloc(type, 0);
    if (obj) {
      obj->cdata = a;
    }
    return (PyObject*) obj;
  }
}

class TraverseExpr : public ExprVisitor<TraverseExpr, int>
{
public:
  int visitPyApply(std::shared_ptr<PyApply> app, visitproc visit, void* arg) {
    Py_VISIT(app->pyobj);
    return 0;
  }
  int visitLet(std::shared_ptr<Let>, visitproc, void*) {
    return 0;
  }
  int visitLocals(std::shared_ptr<Locals>, visitproc, void*) {
    return 0;
  }
};

static int THPExpr_traverse(THPExpr *self, visitproc visit, void *arg)
{
  return self->cdata ? TraverseExpr().visitExpr(self->cdata, visit, arg) : 0;
}

class ClearExpr : public ExprVisitor<ClearExpr>
{
public:
  void visitPyApply(std::shared_ptr<PyApply> app) {
    app->pyobj = nullptr;
  }
  void visitLet(std::shared_ptr<Let>) { }
  void visitLocals(std::shared_ptr<Locals>) { }
};

static int THPExpr_clear(THPExpr *self)
{
  if (self->cdata) ClearExpr().visitExpr(self->cdata);
  return 0;
}

static void THPExpr_dealloc(THPExpr* self)
{
  PyObject_GC_UnTrack(self);
  THPExpr_clear(self);
  self->cdata.~shared_ptr<Expr>();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

class TraverseArg : public ArgVisitor<TraverseArg, int> {
public:
  int visitLocal(std::shared_ptr<Local>, visitproc, void*) { return 0; }
  int visitPyConst(std::shared_ptr<PyConst> c, visitproc visit, void* arg) {
    Py_VISIT(c->pyobj);
    return 0;
  }
};

static int THPArg_traverse(THPArg *self, visitproc visit, void* arg)
{
  return self->cdata ? TraverseArg().visitArg(self->cdata, visit, arg) : 0;
}

class ClearArg : public ArgVisitor<ClearArg>
{
public:
  void visitLocal(std::shared_ptr<Local>) { }
  void visitPyConst(std::shared_ptr<PyConst> c) {
    c->pyobj = nullptr;
  }
};

static int THPArg_clear(THPArg *self)
{
  ClearArg().visitArg(self->cdata);
  return 0;
}

static void THPArg_dealloc(THPArg* self)
{
  PyObject_GC_UnTrack(self);
  THPArg_clear(self);
  self->cdata.~shared_ptr<Arg>();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

typedef PyObject *(*getter)(PyObject *, void *);

/*
PyObject *THPNode_get_inputs(THPNode *self) {
  auto& node = self->cdata;
  auto size = node->inputs.size();
  PyObject* pyInputs(PyTuple_New(size));
  if (!pyInputs) Py_RETURN_NONE;
  for (size_t i = 0; i < size; i++) {
    auto input = PyTuple_New(2);
    PyTuple_SET_ITEM(input, 0, THPNode_Wrap(node->inputs[0].node));
    PyTuple_SET_ITEM(input, 1, PyLong_FromLong(node->inputs[0].output_nr));
    PyTuple_SET_ITEM(pyInputs, i, input);
  }
  return pyInputs;
}

PyObject *THPNode_get_name(THPNode *self) {
  auto& node = self->cdata;
  return THPUtils_packString(node->name());
}
*/

static struct PyGetSetDef THPExpr_properties[] = {
//  {"_inputs", (getter)THPNode_get_inputs, NULL, NULL, NULL},
//  {"_name", (getter)THPNode_get_name, NULL, NULL, NULL},
  {NULL}
};

static struct PyGetSetDef THPArg_properties[] = {
  {NULL}
};

static PyObject* THPExpr_str(THPExpr *self) {
  std::stringstream ss;
  printExpr(self->cdata, ss);
  return THPUtils_packString(ss.str());
}

PyTypeObject THPExprType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._ExprBase",                  /* tp_name */
  sizeof(THPExpr),                       /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPExpr_dealloc,           /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  (reprfunc)THPExpr_str,                 /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */
  NULL,                                  /* tp_doc */
  (traverseproc)THPExpr_traverse,        /* tp_traverse */
  (inquiry)THPExpr_clear,                /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  0,                                     /* tp_methods */
  0,                                     /* tp_members */
  THPExpr_properties,                    /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  // TODO: add me, seems reasonable
  0                                      /* tp_new */
};

PyTypeObject THPArgType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._ArgBase",                  /* tp_name */
  sizeof(THPArg),                       /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPArg_dealloc,           /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */
  NULL,                                  /* tp_doc */
  (traverseproc)THPArg_traverse,        /* tp_traverse */
  (inquiry)THPArg_clear,                /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  0,                                     /* tp_methods */
  0,                                     /* tp_members */
  THPArg_properties,                    /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  // TODO: add me, seems reasonable
  0                                      /* tp_new */
};

bool THPIR_initModule(PyObject *module) {
  if (PyType_Ready(&THPExprType) < 0)
    return false;
  if (PyType_Ready(&THPArgType) < 0)
    return false;
  Py_INCREF(&THPExprType);
  Py_INCREF(&THPArgType);
  PyModule_AddObject(module, "_ExprBase", (PyObject *)&THPExprType);
  PyModule_AddObject(module, "_ArgBase", (PyObject *)&THPArgType);
  return true;
}
