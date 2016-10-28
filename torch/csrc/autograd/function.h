#ifndef THP_FUNCTION_H
#define THP_FUNCTION_H

struct THPFunction;

struct THPFunctionPtr: public THPObjectPtr {
    THPFunctionPtr(): THPObjectPtr(nullptr), output_nr(-1) {};

    THPFunctionPtr(PyObject *fn, int output_nr):
        THPObjectPtr(fn), output_nr(output_nr) {};

    THPFunctionPtr(THPFunction *fn, int output_nr):
        THPObjectPtr((PyObject*)fn), output_nr(output_nr) {};

    THPFunctionPtr(THPFunctionPtr &&other):
        THPObjectPtr(std::move(other)), output_nr(other.output_nr) {}

    THPPointer& operator =(THPFunctionPtr &&other) {
        output_nr = other.output_nr;
        THPObjectPtr::operator=(std::move(other));
        return *this;
    }

    int output_nr;
};

// (class, gpu id, sizes)
using output_info_type = std::tuple<PyObject *, int, std::vector<long>>;

struct THPFunction {
    PyObject_HEAD

    PyObject *needs_input_grad;
    PyObject *saved_variables;
    PyObject *backward_hooks;

    PyObject *to_save;
    PyObject *shared_pairs;
    PyObject *non_differentiable;
    PyObject *dirty_tensors;

    THPFunctionPtr *previous_functions;
    std::vector<output_info_type> *output_info;
    int num_inputs;
    int num_outputs;
    char requires_grad;
    char has_freed_buffers;
};

bool THPFunction_initModule(PyObject *module);
extern PyObject *THPFunctionClass;

#define THPFunction_Check(obj) PyObject_IsInstance(obj, THPFunctionClass)

#endif
