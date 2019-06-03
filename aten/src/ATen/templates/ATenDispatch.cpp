#include <ATen/core/ATenDispatch.h>

#include <unordered_map>
#include <ATen/Context.h>

namespace at {

ATenDispatch & globalATenDispatch() {
  static ATenDispatch singleton;
  return singleton;
}

void ATenDispatch::initCuda() {
  globalContext().lazyInitCUDA();
}

int64_t ATenDispatch::getSchemaId(std::string schema) {
  static std::unordered_map<std::string, int64_t> schema_to_id = {
    ${schema_to_id_pairs}
  };
  return schema_to_id[schema];
}

void** ATenDispatch::getFunctionTable(Backend backend) {
  static void* function_table[static_cast<int64_t>(Backend::NumOptions)][${function_count}];
  return function_table[static_cast<int64_t>(backend)];
}
void** ATenDispatch::getWrapperTable() {
  static void* wrapper_table[${function_count}];
  return wrapper_table;
}

} // namespace at
