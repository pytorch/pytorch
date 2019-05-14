#include <ATen/TypeDefault.h>

namespace at {

static std::map<std::string, int64_t> schema_to_id = {
  ${schema_to_id_pairs}
};
static void* function_table[static_cast<int64_t>(Backend::NumOptions)][${function_count}];
static void* wrapper_table[${function_count}];

int64_t _register_op(Backend backend, const char* schema, void* fn) {
  auto id = schema_to_id[schema];
  if (backend == Backend::Undefined) {
    for (int i = 0; i < static_cast<int>(Backend::NumOptions); i++) {
      function_table[i][id] = fn;
    }
  }
  function_table[static_cast<int64_t>(backend)][id] = fn;
  return id;
}

int64_t _register_variable_wrapper(const char* schema, void* fn) {
  auto id = schema_to_id[schema];
  wrapper_table[id] = fn;
  return id;
}

void* get_op(Backend backend, int64_t id) {
  if (function_table[static_cast<int64_t>(backend)][id] == nullptr) {
    AT_ERROR("asdf");
  }
  return function_table[static_cast<int64_t>(backend)][id];
}

void* get_variable_wrapper(int64_t id) {
  if (wrapper_table[id] == nullptr) {
    AT_ERROR("asdf");
  }
  return wrapper_table[id];
}
} // namespace at
