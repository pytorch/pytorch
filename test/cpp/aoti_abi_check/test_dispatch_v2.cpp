#include <gtest/gtest.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/util/Exception.h>

#define DEFINE_ITEM(TYPE, SCALARTYPE) ScalarType::SCALARTYPE,

#define TEST_DISPATCH_V2(NAME, EXPECTEDCOUNT, ...)                       \
  TEST(TestThoDispatchV2, NAME) {                                        \
    using torch::headeronly::ScalarType;                                 \
    using torch::headeronly::impl::ScalarTypeToCPPTypeT;                 \
    int8_t total_count = 0;                                              \
    int8_t count = 0;                                                    \
    int8_t default_count = 0;                                            \
    for (ScalarType t :                                                  \
         {AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ITEM)}) { \
      total_count++;                                                     \
      try {                                                              \
        THO_DISPATCH_V2(                                                 \
            t,                                                           \
            "test_tho_dispatch_v2",                                      \
            [&] {                                                        \
              count++;                                                   \
              scalar_t tmp;                                              \
              (void)tmp;                                                 \
            },                                                           \
            __VA_ARGS__);                                                \
      } catch (...) {                                                    \
        default_count++; /* counts mismatches */                         \
      }                                                                  \
    }                                                                    \
    EXPECT_EQ(count, EXPECTEDCOUNT);                                     \
    EXPECT_EQ(default_count + count, total_count);                       \
  }

TEST_DISPATCH_V2(AT_FLOAT8_TYPES_, 5, AT_FLOAT8_TYPES);
TEST_DISPATCH_V2(AT_INTEGRAL_TYPES_, 5, AT_INTEGRAL_TYPES);
TEST_DISPATCH_V2(AT_FLOATING_TYPES_, 2, AT_FLOATING_TYPES);
TEST_DISPATCH_V2(AT_BAREBONES_UNSIGNED_TYPES_, 3, AT_BAREBONES_UNSIGNED_TYPES);
TEST_DISPATCH_V2(AT_INTEGRAL_TYPES_V2_, 8, AT_INTEGRAL_TYPES_V2);
TEST_DISPATCH_V2(AT_COMPLEX_TYPES_, 2, AT_COMPLEX_TYPES);
TEST_DISPATCH_V2(AT_QINT_TYPES_, 3, AT_QINT_TYPES);
TEST_DISPATCH_V2(AT_ALL_TYPES_, 7, AT_ALL_TYPES);
TEST_DISPATCH_V2(AT_ALL_TYPES_AND_COMPLEX_, 9, AT_ALL_TYPES_AND_COMPLEX);

#undef DEFINE_ITEM
