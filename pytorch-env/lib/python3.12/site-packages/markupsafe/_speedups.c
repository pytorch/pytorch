#include <Python.h>

#define GET_DELTA(inp, inp_end, delta) \
	while (inp < inp_end) { \
		switch (*inp++) { \
		case '"': \
		case '\'': \
		case '&': \
			delta += 4; \
			break; \
		case '<': \
		case '>': \
			delta += 3; \
			break; \
		} \
	}

#define DO_ESCAPE(inp, inp_end, outp) \
	{ \
		Py_ssize_t ncopy = 0; \
		while (inp < inp_end) { \
			switch (*inp) { \
			case '"': \
				memcpy(outp, inp-ncopy, sizeof(*outp)*ncopy); \
				outp += ncopy; ncopy = 0; \
				*outp++ = '&'; \
				*outp++ = '#'; \
				*outp++ = '3'; \
				*outp++ = '4'; \
				*outp++ = ';'; \
				break; \
			case '\'': \
				memcpy(outp, inp-ncopy, sizeof(*outp)*ncopy); \
				outp += ncopy; ncopy = 0; \
				*outp++ = '&'; \
				*outp++ = '#'; \
				*outp++ = '3'; \
				*outp++ = '9'; \
				*outp++ = ';'; \
				break; \
			case '&': \
				memcpy(outp, inp-ncopy, sizeof(*outp)*ncopy); \
				outp += ncopy; ncopy = 0; \
				*outp++ = '&'; \
				*outp++ = 'a'; \
				*outp++ = 'm'; \
				*outp++ = 'p'; \
				*outp++ = ';'; \
				break; \
			case '<': \
				memcpy(outp, inp-ncopy, sizeof(*outp)*ncopy); \
				outp += ncopy; ncopy = 0; \
				*outp++ = '&'; \
				*outp++ = 'l'; \
				*outp++ = 't'; \
				*outp++ = ';'; \
				break; \
			case '>': \
				memcpy(outp, inp-ncopy, sizeof(*outp)*ncopy); \
				outp += ncopy; ncopy = 0; \
				*outp++ = '&'; \
				*outp++ = 'g'; \
				*outp++ = 't'; \
				*outp++ = ';'; \
				break; \
			default: \
				ncopy++; \
			} \
			inp++; \
		} \
		memcpy(outp, inp-ncopy, sizeof(*outp)*ncopy); \
	}

static PyObject*
escape_unicode_kind1(PyUnicodeObject *in)
{
	Py_UCS1 *inp = PyUnicode_1BYTE_DATA(in);
	Py_UCS1 *inp_end = inp + PyUnicode_GET_LENGTH(in);
	Py_UCS1 *outp;
	PyObject *out;
	Py_ssize_t delta = 0;

	GET_DELTA(inp, inp_end, delta);
	if (!delta) {
		Py_INCREF(in);
		return (PyObject*)in;
	}

	out = PyUnicode_New(PyUnicode_GET_LENGTH(in) + delta,
						PyUnicode_IS_ASCII(in) ? 127 : 255);
	if (!out)
		return NULL;

	inp = PyUnicode_1BYTE_DATA(in);
	outp = PyUnicode_1BYTE_DATA(out);
	DO_ESCAPE(inp, inp_end, outp);
	return out;
}

static PyObject*
escape_unicode_kind2(PyUnicodeObject *in)
{
	Py_UCS2 *inp = PyUnicode_2BYTE_DATA(in);
	Py_UCS2 *inp_end = inp + PyUnicode_GET_LENGTH(in);
	Py_UCS2 *outp;
	PyObject *out;
	Py_ssize_t delta = 0;

	GET_DELTA(inp, inp_end, delta);
	if (!delta) {
		Py_INCREF(in);
		return (PyObject*)in;
	}

	out = PyUnicode_New(PyUnicode_GET_LENGTH(in) + delta, 65535);
	if (!out)
		return NULL;

	inp = PyUnicode_2BYTE_DATA(in);
	outp = PyUnicode_2BYTE_DATA(out);
	DO_ESCAPE(inp, inp_end, outp);
	return out;
}


static PyObject*
escape_unicode_kind4(PyUnicodeObject *in)
{
	Py_UCS4 *inp = PyUnicode_4BYTE_DATA(in);
	Py_UCS4 *inp_end = inp + PyUnicode_GET_LENGTH(in);
	Py_UCS4 *outp;
	PyObject *out;
	Py_ssize_t delta = 0;

	GET_DELTA(inp, inp_end, delta);
	if (!delta) {
		Py_INCREF(in);
		return (PyObject*)in;
	}

	out = PyUnicode_New(PyUnicode_GET_LENGTH(in) + delta, 1114111);
	if (!out)
		return NULL;

	inp = PyUnicode_4BYTE_DATA(in);
	outp = PyUnicode_4BYTE_DATA(out);
	DO_ESCAPE(inp, inp_end, outp);
	return out;
}

static PyObject*
escape_unicode(PyObject *self, PyObject *s)
{
	if (!PyUnicode_Check(s))
		return NULL;

    // This check is no longer needed in Python 3.12.
	if (PyUnicode_READY(s))
		return NULL;

	switch (PyUnicode_KIND(s)) {
	case PyUnicode_1BYTE_KIND:
		return escape_unicode_kind1((PyUnicodeObject*) s);
	case PyUnicode_2BYTE_KIND:
		return escape_unicode_kind2((PyUnicodeObject*) s);
	case PyUnicode_4BYTE_KIND:
		return escape_unicode_kind4((PyUnicodeObject*) s);
	}
	assert(0);  /* shouldn't happen */
	return NULL;
}

static PyMethodDef module_methods[] = {
	{"_escape_inner", (PyCFunction)escape_unicode, METH_O, NULL},
	{NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef module_definition = {
	PyModuleDef_HEAD_INIT,
	"markupsafe._speedups",
	NULL,
	-1,
	module_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit__speedups(void)
{
	PyObject *m = PyModule_Create(&module_definition);

	if (m == NULL) {
		return NULL;
	}

	#ifdef Py_GIL_DISABLED
	PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
	#endif

	return m;
}
