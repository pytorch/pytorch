#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct _PytorchRecordFunctionState;
typedef struct _PytorchRecordFunctionState _PytorchRecordFunctionState;

_PytorchRecordFunctionState* _pytorch_record_function_enter(const char* name);
void _pytorch_record_function_exit(_PytorchRecordFunctionState* state);

void _compiled_region_enter();
void _compiled_region_exit();

#ifdef __cplusplus
} // extern "C"
#endif
