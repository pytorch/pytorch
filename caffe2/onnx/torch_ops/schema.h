#pragma once

#include "./constants.h"
#include "onnx/defs/schema.h"

#define ONNX_PYTORCH_OPERATOR_SET_SCHEMA(name, ver, impl) \
  ONNX_OPERATOR_SET_SCHEMA_EX(                            \
      name, PyTorch, AI_ONNX_PYTORCH_DOMAIN, ver, false, impl)
