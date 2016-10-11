#ifndef THP_ENGINE_H
#define THP_ENGINE_H

struct THPEngine {
    PyObject_HEAD
};

bool THPEngine_initModule(PyObject *module);

#endif
