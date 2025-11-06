#include <gtest/gtest.h>

#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/Dispatch_v2.h>

// MY_PRIVATE_CHECK_SELECTIVE_BUILD is a prelude to case block. For
// testing, we do nothing:
#define MY_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type) /* empty */

#define MY_PRIVATE_CASE_TYPE_USING_HINT(...) \
  THO_PRIVATE_CASE_TYPE_USING_HINT_TMPL(     \
      MY_PRIVATE_CHECK_SELECTIVE_BUILD, __VA_ARGS__)

#define MY_DISPATCH_CASE(...) \
  THO_DISPATCH_CASE_TMPL(MY_PRIVATE_CASE_TYPE_USING_HINT, __VA_ARGS__)

// MY_RECORD_KERNEL_FUNCTION_DTYPE is a prelude to switch
// statement. For testing, we just avoid unused variable warning:
#define MY_RECORD_KERNEL_FUNCTION_DTYPE(DISPATCHNAME, ENUMTYPE) \
  (void)DISPATCHNAME

// MY_CHECK_NOT_IMPLEMENTED is called in switch default block. For
// testing, we count case mismatches:
#define MY_CHECK_NOT_IMPLEMENTED(...) default_count++

#define MY_DISPATCH_SWITCH(...) \
  THO_DISPATCH_SWITCH_TMPL(     \
      MY_RECORD_KERNEL_FUNCTION_DTYPE, MY_CHECK_NOT_IMPLEMENTED, __VA_ARGS__)

// MY_CASE_FUNCTION is called in a case block. For testing, we count
// case matches and ensure that scalar_t/index_t type is defined:
#define MY_CASE_FUNCTION \
  [&] {                  \
    count++;             \
    scalar_t tmp;        \
    (void)tmp;           \
  }
#define MY_INDEX_CASE_FUNCTION \
  [&] {                        \
    count++;                   \
    index_t tmp;               \
    (void)tmp;                 \
  }

#define DEFINE_ITEM(TYPE, SCALARTYPE) ScalarType::SCALARTYPE,

#define MY_DISPATCH_V2(TYPE, NAME, BODY, ...) \
  THO_DISPATCH_V2_TMPL(                       \
      MY_DISPATCH_SWITCH,                     \
      MY_DISPATCH_CASE,                       \
      TYPE,                                   \
      NAME,                                   \
      AT_WRAP(BODY),                          \
      __VA_ARGS__)

#define TEST_DISPATCH_V2(NAME, EXPECTEDCOUNT, ...)                             \
  TEST(TestDispatchV2, NAME) {                                                 \
    using torch::headeronly::ScalarType;                                       \
    using torch::headeronly::impl::ScalarTypeToCPPTypeT;                       \
    int8_t total_count = 0;                                                    \
    int8_t count = 0;                                                          \
    int8_t default_count = 0;                                                  \
    for (ScalarType t :                                                        \
         {AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ITEM)}) {       \
      total_count++;                                                           \
      MY_DISPATCH_V2(t, "test_my_dispatch_v2", MY_CASE_FUNCTION, __VA_ARGS__); \
    }                                                                          \
    EXPECT_EQ(count, EXPECTEDCOUNT);                                           \
    EXPECT_EQ(default_count + count, total_count);                             \
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
