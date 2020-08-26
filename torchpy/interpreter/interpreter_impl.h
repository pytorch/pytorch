#pragma once

static void run_some_python(const char* code);
static void teardown();
static void run_python_file(const char* code);

#define FOREACH_INTERFACE_FUNCTION(_) \
  _(run_some_python)                  \
  _(teardown)                         \
  _(run_python_file)

struct InterpreterImpl {
#define DEFINE_POINTER(func) decltype(&::func) func;
  FOREACH_INTERFACE_FUNCTION(DEFINE_POINTER)
#undef DEFINE_POINTER
};
