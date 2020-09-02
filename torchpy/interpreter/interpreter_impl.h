#pragma once
#include <ATen/ATen.h>

static size_t load_model(const char* model_file);
static at::Tensor forward_model(size_t model_id, at::Tensor input);
static void run_some_python(const char* code);
static void teardown();
static void run_python_file(const char* code);

#define FOREACH_INTERFACE_FUNCTION(_) \
  _(load_model)                       \
  _(forward_model)                    \
  _(run_some_python)                  \
  _(teardown)                         \
  _(run_python_file)

struct InterpreterImpl {
#define DEFINE_POINTER(func) decltype(&::func) func;
  FOREACH_INTERFACE_FUNCTION(DEFINE_POINTER)
#undef DEFINE_POINTER
};

static std::atomic<size_t> s_interpreter_id;