#include <gtest/gtest.h>

#include <string>

#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace cpp_test {

TEST(MaybeRefTest, BasicTest) {
  using StringRef = lazy_tensors::util::MaybeRef<std::string>;
  std::string storage("String storage");
  StringRef ref_storage(storage);
  EXPECT_FALSE(ref_storage.is_stored());
  EXPECT_EQ(*ref_storage, storage);

  StringRef eff_storage(std::string("Vanishing"));
  EXPECT_TRUE(eff_storage.is_stored());
  EXPECT_EQ(*eff_storage, "Vanishing");
}

}  // namespace cpp_test
}  // namespace torch_lazy_tensors
