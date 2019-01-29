#pragma once

#include <ATen/core/dispatch/Dispatcher.h>

/**
 * Macro for defining an operator schema.  Every operator schema must
 * invoke C10_DECLARE_OP_SCHEMA in a header and C10_DEFINE_OP_SCHEMA in one (!)
 * cpp file.  Internally, this arranges for the dispatch table for
 * the operator to be created.
 */
#define C10_DECLARE_OP_SCHEMA(Name)                                             \
  CAFFE2_API const c10::OperatorHandle& Name();                                 \

#define C10_DEFINE_OP_SCHEMA(Name, Schema)                                      \
  C10_EXPORT const c10::OperatorHandle& Name() {                                \
    static c10::OperatorHandle singleton =                                      \
        c10::Dispatcher::singleton().registerSchema(Schema);                    \
    return singleton;                                                           \
  }
