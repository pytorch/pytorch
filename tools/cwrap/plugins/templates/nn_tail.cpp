namespace torch { namespace nn {

static PyTypeObject thnn_type;

void init_$short_name(PyObject* c_module) {
  ((PyObject*)&thnn_type)->ob_refcnt = 1;
  thnn_type.tp_flags = Py_TPFLAGS_DEFAULT;
  thnn_type.tp_methods = module_methods;
  thnn_type.tp_name = "torch._C.$short_name";
  if (PyType_Ready(&thnn_type) < 0) {
    throw python_error();
  }

  PyObject* type_obj = (PyObject*)&thnn_type;
  Py_INCREF(type_obj);
  if (PyModule_AddObject(c_module, "$short_name", type_obj) < 0) {
    throw python_error();
  }
}

}}  // namespace torch::nn
