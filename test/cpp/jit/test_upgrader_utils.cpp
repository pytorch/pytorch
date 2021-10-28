#include <torch/csrc/jit/operator_upgraders/version_map.h>
#include <torch/csrc/jit/operator_upgraders/utils.h>
#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/operator_upgraders/utils.h>

namespace torch {
namespace jit {

TEST(UpgraderUtils, FindCorrectUpgrader) {

    UpgraderDB dummy_entry = {
        {4, "foo__0_3"},
        {8, "foo__4_7"},
    };

    auto upgrader = findUpgrader(dummy_entry, 6);
    EXPECT_EQ(upgrader, "foo__4_7");
}

} // namespace jit
} // namespace torch
