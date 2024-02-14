#include <torch/csrc/jit/runtime/update_disable_alias_db.h>

namespace torch::jit {

thread_local bool kDisableAliasDb = false;
void setDisableAliasDb(bool o) {
  kDisableAliasDb = o;
}
bool getDisableAliasDb() {
  return kDisableAliasDb;
}

} // namespace torch::jit
