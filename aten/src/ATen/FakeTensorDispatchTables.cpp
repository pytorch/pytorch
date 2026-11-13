#include <ATen/FakeTensorDispatchTables.h>

#include <c10/util/LeftRight.h>

#include <unordered_map>

namespace at::impl {
namespace {

// The categories each registered operator is in, as a fakeDispatchCategoryBit
// mask; operators in none are absent
using FakeDispatchTables = std::unordered_map<c10::OperatorName, uint32_t>;

c10::LeftRight<FakeDispatchTables>& fakeDispatchTables() {
  static c10::LeftRight<FakeDispatchTables> tables;
  return tables;
}

} // namespace

void fakeDispatchTableAdd(
    FakeDispatchCategory category,
    const c10::OperatorName& name) {
  fakeDispatchTables().write([&](FakeDispatchTables& t) {
    t[name] |= fakeDispatchCategoryBit(category);
  });
}

void fakeDispatchTableRemove(
    FakeDispatchCategory category,
    const c10::OperatorName& name) {
  fakeDispatchTables().write([&](FakeDispatchTables& t) {
    const auto it = t.find(name);
    if (it != t.end() &&
        (it->second &= ~fakeDispatchCategoryBit(category)) == 0) {
      t.erase(it);
    }
  });
}

bool fakeDispatchTableContains(
    FakeDispatchCategory category,
    const c10::OperatorName& name) {
  return fakeDispatchCategoriesHave(fakeDispatchTableCategories(name), category);
}

uint32_t fakeDispatchTableCategories(const c10::OperatorName& name) {
  return fakeDispatchTables().read([&](const FakeDispatchTables& t) {
    const auto it = t.find(name);
    return it == t.end() ? 0 : it->second;
  });
}

} // namespace at::impl
