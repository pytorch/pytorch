#pragma once
#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

struct _PytorchRecordFunctionState;
typedef struct _PytorchRecordFunctionState _PytorchRecordFunctionState;

_PytorchRecordFunctionState* _pytorch_record_function_enter(const char* name);
_PytorchRecordFunctionState* _pytorch_record_function_enter_with_context(
    const char* name,
    const char* context);
void _pytorch_record_function_exit(_PytorchRecordFunctionState* state);

bool eval_frame_callback_enabled_get();
void eval_frame_callback_enabled_set(bool enabled);

#ifdef __cplusplus
} // extern "C"
#endif
