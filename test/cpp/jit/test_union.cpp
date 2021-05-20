#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include "torch/csrc/jit/ir/ir.h"
#include "torch/csrc/jit/ir/irparser.h"

namespace torch {
namespace jit {


TEST(UnionTypeTest, SubtypingWithOptional) {
    // None
    const TypePtr none = NoneType::get();

    // Optional[int]
    const TypePtr opt = OptionalType::create(IntType::get());

    // Union[int, None]
    const TypePtr union1 = UnionType::create(std::vector<TypePtr>{IntType::get(), NoneType::get()});

    // Union[int, str, None]
    const TypePtr union2 = UnionType::create(std::vector<TypePtr>{IntType::get(), StringType::get(), NoneType::get()});

    // Union[int, str, List[str]]
    const TypePtr union3 = UnionType::create(std::vector<TypePtr>{IntType::get(), StringType::get(), ListType::ofStrings()});

    ASSERT_TRUE(none->isSubtypeOf(opt));
    ASSERT_TRUE(none->isSubtypeOf(union1));
    ASSERT_TRUE(none->isSubtypeOf(union2));
    ASSERT_FALSE(none->isSubtypeOf(union3));

    ASSERT_FALSE(opt->isSubtypeOf(none));
    ASSERT_TRUE(opt->isSubtypeOf(union1));
    ASSERT_TRUE(opt->isSubtypeOf(union2));
    ASSERT_FALSE(opt->isSubtypeOf(union3));

    ASSERT_FALSE(union1->isSubtypeOf(none));
    ASSERT_TRUE(union1->isSubtypeOf(opt));
    ASSERT_TRUE(union1->isSubtypeOf(union2));
    ASSERT_FALSE(union1->isSubtypeOf(union3));

    ASSERT_FALSE(union2->isSubtypeOf(union1));
}

} // namespace torch
} // namespace jit
