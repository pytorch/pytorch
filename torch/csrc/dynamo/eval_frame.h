#pragma once
#include <Python.h>

extern "C" {
PyObject* torch_c_dynamo_eval_frame_init(void);
extern bool is_dynamo_compiling;
}
