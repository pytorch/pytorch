#pragma once

#include <ATen/core/dispatch/Dispatcher.h>

// TODO Better error message when this definition is missing

/**
 * Macro for defining an operator schema.  Every user-defined OpSchemaDef struct must
 * invoke this macro on it.  Internally, this arranges for the dispatch table for
 * the operator to be created.
 */
#define C10_DEFINE_OP_SCHEMA(OpSchemaDef)                                         \
  template<>                                                                      \
  C10_EXPORT c10::DispatchTable<OpSchemaDef>& c10_dispatch_table<OpSchemaDef>() { \
    static c10::DispatchTable<OpSchemaDef> singleton;                             \
    return singleton;                                                             \
  }
// TODO Also register unboxed calling API here
