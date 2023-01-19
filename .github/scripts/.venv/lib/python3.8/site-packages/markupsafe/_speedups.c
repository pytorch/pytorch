#include <Python.h>

static PyObject* markup;

static int
init_constants(void)
{
	PyObject *module;

	/* import markup type so that we can mark the return value */
	module = PyImport_ImportModule("markupsafe");
	if (!module)
		return 0;
	markup = PyObject_GetAttrString(module, "Markup");
	Py_DECREF(module);

	return 1;
}

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
escape_unicode(PyUnicodeObject *in)
{
	if (PyUnicode_READY(in))
		return NULL;

	switch (PyUnicode_KIND(in)) {
	case PyUnicode_1BYTE_KIND:
		return escape_unicode_kind1(in);
	case PyUnicode_2BYTE_KIND:
		return escape_unicode_kind2(in);
	case PyUnicode_4BYTE_KIND:
		return escape_unicode_kind4(in);
	}
	assert(0);  /* shouldn't happen */
	return NULL;
}

static PyObject*
escape(PyObject *self, PyObject *text)
{
	static PyObject *id_html;
	PyObject *s = NULL, *rv = NULL, *html;

	if (id_html == NULL) {
		id_html = PyUnicode_InternFromString("__html__");
		if (id_html == NULL) {
			return NULL;
		}
	}

	/* we don't have to escape integers, bools or floats */
	if (PyLong_CheckExact(text) ||
		PyFloat_CheckExact(text) || PyBool_Check(text) ||
		text == Py_None)
		return PyObject_CallFunctionObjArgs(markup, text, NULL);

	/* if the object has an __html__ method that performs the escaping */
	html = PyObject_GetAttr(text ,id_html);
	if (html) {
		s = PyObject_CallObject(html, NULL);
		Py_DECREF(html);
		if (s == NULL) {
			return NULL;
		}
		/* Convert to Markup object */
		rv = PyObject_CallFunctionObjArgs(markup, (PyObject*)s, NULL);
		Py_DECREF(s);
		return rv;
	}

	/* otherwise make the object unicode if it isn't, then escape */
	PyErr_Clear();
	if (!PyUnicode_Check(text)) {
		PyObject *unicode = PyObject_Str(text);
		if (!unicode)
			return NULL;
		s = escape_unicode((PyUnicodeObject*)unicode);
		Py_DECREF(unicode);
	}
	else
		s = escape_unicode((PyUnicodeObject*)text);

	/* convert the unicode string into a markup object. */
	rv = PyObject_CallFunctionObjArgs(markup, (PyObject*)s, NULL);
	Py_DECREF(s);
	return rv;
}


static PyObject*
escape_silent(PyObject *self, PyObject *text)
{
	if (text != Py_None)
		return escape(self, text);
	return PyObject_CallFunctionObjArgs(markup, NULL);
}


static PyObject*
soft_str(PyObject *self, PyObject *s)
{
	if (!PyUnicode_Check(s))
		return PyObject_Str(s);
	Py_INCREF(s);
	return s;
}


static PyMethodDef module_methods[] = {
	{
		"escape",
		(PyCFunction)escape,
		METH_O,
		"Replace the characters ``&``, ``<``, ``>``, ``'``, and ``\"`` in"
		" the string with HTML-safe sequences. Use this if you need to display"
		" text that might contain such characters in HTML.\n\n"
		"If the object has an ``__html__`` method, it is called and the"
		" return value is assumed to already be safe for HTML.\n\n"
		":param s: An object to be converted to a string and escaped.\n"
		":return: A :class:`Markup` string with the escaped text.\n"
	},
	{
		"escape_silent",
		(PyCFunction)escape_silent,
		METH_O,
		"Like :func:`escape` but treats ``None`` as the empty string."
		" Useful with optional values, as otherwise you get the string"
		" ``'None'`` when the value is ``None``.\n\n"
		">>> escape(None)\n"
		"Markup('None')\n"
		">>> escape_silent(None)\n"
		"Markup('')\n"
	},
	{
		"soft_str",
		(PyCFunction)soft_str,
		METH_O,
		"Convert an object to a string if it isn't already. This preserves"
		" a :class:`Markup` string rather than converting it back to a basic"
		" string, so it will still be marked as safe and won't be escaped"
		" again.\n\n"
		">>> value = escape(\"<User 1>\")\n"
		">>> value\n"
		"Markup('&lt;User 1&gt;')\n"
		">>> escape(str(value))\n"
		"Markup('&amp;lt;User 1&amp;gt;')\n"
		">>> escape(soft_str(value))\n"
		"Markup('&lt;User 1&gt;')\n"
	},
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
	if (!init_constants())
		return NULL;

	return PyModule_Create(&module_definition);
}
