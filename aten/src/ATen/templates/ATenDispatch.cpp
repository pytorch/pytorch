#include <ATen/core/ATenDispatch.h>
#include <ATen/Context.h>

#include <unordered_map>

namespace at {

static void* function_table[static_cast<int64_t>(Backend::NumOptions)][${function_count}];
static void* wrapper_table[${function_count}];

int64_t get_schema_id(std::string schema) {
  static std::unordered_map<std::string, int64_t> schema_to_id = {
    ${schema_to_id_pairs}
  };
  return schema_to_id[schema];
}

int64_t _register_op(Backend backend, const char* schema, void* fn) {
  auto id = get_schema_id(schema);
  function_table[static_cast<int64_t>(backend)][id] = fn;
  return id;
}

int64_t _register_variable_wrapper(const char* schema, void* fn) {
  auto id = get_schema_id(schema);
  wrapper_table[id] = fn;
  return id;
}

void* get_op(Backend backend, int64_t id) {
  if (function_table[static_cast<int64_t>(backend)][id] == nullptr) {
    if (function_table[static_cast<int64_t>(Backend::Undefined)][id] == nullptr) {
      AT_ERROR("asdf");
    }
    function_table[static_cast<int64_t>(backend)][id] = function_table[static_cast<int64_t>(Backend::Undefined)][id];
  }
  if (backend == Backend::CUDA) {
    globalContext().lazyInitCUDA();
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
