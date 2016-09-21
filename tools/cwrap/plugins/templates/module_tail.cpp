
#if PY_MAJOR_VERSION != 2
static struct PyModuleDef module_def = {
   PyModuleDef_HEAD_INIT,
   "$full_name",
   NULL,
   -1,
   module_methods
};
#endif

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init$short_name()
#else
PyMODINIT_FUNC PyInit_$short_name()
#endif
{
#if PY_MAJOR_VERSION == 2
#define ASSERT_TRUE(cmd) if (!(cmd)) {PyErr_SetString(PyExc_ImportError, "initialization error"); return;}
#else
#define ASSERT_TRUE(cmd) if (!(cmd)) return NULL
#endif
  PyObject *module;

#if PY_MAJOR_VERSION == 2
  ASSERT_TRUE(module = Py_InitModule("$full_name", module_methods));
#else
  ASSERT_TRUE(module = PyModule_Create(&module_def));
#endif

#if PY_MAJOR_VERSION != 2
  return module;
#endif

#undef ASSERT_TRUE
}
