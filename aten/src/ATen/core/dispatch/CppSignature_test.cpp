#include <ATen/core/dispatch/CppSignature.h>
#include <gtest/gtest.h>
#include <string>

using c10::impl::CppSignature;

namespace {

TEST(CppSignatureTest, given_equalSignature_then_areEqual) {
    EXPECT_EQ(CppSignature::make<void()>(), CppSignature::make<void()>());
    EXPECT_EQ(CppSignature::make<int64_t(std::string, int64_t)>(), CppSignature::make<int64_t(std::string, int64_t)>());
}

TEST(CppSignatureTest, given_differentSignature_then_areDifferent) {
    EXPECT_NE(CppSignature::make<void()>(), CppSignature::make<int64_t()>());
    EXPECT_NE(CppSignature::make<int64_t(std::string)>(), CppSignature::make<int64_t(std::string, int64_t)>());
    EXPECT_NE(CppSignature::make<std::string(std::string)>(), CppSignature::make<int64_t(std::string)>());
}

TEST(CppSignatureTest, given_equalFunctorAndFunction_then_areEqual) {
    struct Functor final {
        int64_t operator()(std::string) {return 0;}
    };
    EXPECT_EQ(CppSignature::make<Functor>(), CppSignature::make<int64_t(std::string)>());
}

TEST(CppSignatureTest, given_differentFunctorAndFunction_then_areDifferent) {
    struct Functor final {
        int64_t operator()(std::string) {return 0;}
    };
    EXPECT_NE(CppSignature::make<Functor>(), CppSignature::make<int64_t(std::string, int64_t)>());
}

TEST(CppSignatureTest, given_cppSignature_then_canQueryNameWithoutCrashing) {
    CppSignature::make<void(int64_t, const int64_t&)>().name();
}

}
