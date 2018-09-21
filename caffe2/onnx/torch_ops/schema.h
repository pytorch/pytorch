#pragma once

#include "onnx/defs/schema.h"
#include "./constants.h"

#define ONNX_PYTORCH_OPERATOR_SET_SCHEMA(name, ver, impl)               \
    ONNX_OPERATOR_SET_SCHEMA_EX(name, PyTorch, AI_ONNX_PYTORCH_DOMAIN, ver, true, impl)
