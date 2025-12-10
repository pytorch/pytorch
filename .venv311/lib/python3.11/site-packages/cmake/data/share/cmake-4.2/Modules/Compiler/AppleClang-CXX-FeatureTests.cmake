
# No known reference for AppleClang versions.
# Generic reference: http://clang.llvm.org/cxx_status.html
# http://clang.llvm.org/docs/LanguageExtensions.html

# Note: CXX compiler in Xcode 4.3 does not set __apple_build_version__ and so is
# not recognized as AppleClang.
# Xcode_43 - Apple clang version 3.1 (tags/Apple/clang-318.0.61) (based on LLVM 3.1svn)
# Xcode_44 - Apple clang version 4.0 (tags/Apple/clang-421.0.60) (based on LLVM 3.1svn)
# Xcode_45 - Apple clang version 4.1 (tags/Apple/clang-421.11.66) (based on LLVM 3.1svn)
# Xcode_46 - Apple LLVM version 4.2 (clang-425.0.28) (based on LLVM 3.2svn)
# Xcode_50 - Apple LLVM version 5.0 (clang-500.2.79) (based on LLVM 3.3svn)
# Xcode_51 - Apple LLVM version 5.1 (clang-503.0.38) (based on LLVM 3.4svn)
# Xcode_60 - Apple LLVM version 6.0 (clang-600.0.51) (based on LLVM 3.5svn)
# Xcode_61 - Apple LLVM version 6.0 (clang-600.0.56) (based on LLVM 3.5svn)

# There is some non-correspondence. __has_feature(cxx_user_literals) is
# false for AppleClang 4.0 and 4.1, although it is reported as
# supported in the reference link for Clang 3.1.  The compiler does not pass
# the CompileFeatures/cxx_user_literals.cpp test.
# cxx_attributes is listed as not supported until Clang 3.3. It works without
# warning with AppleClang 5.0, but issues a gcc-compat warning for
# AppleClang 4.0-4.2.
# cxx_alignof and cxx_alignas tests work for early AppleClang versions, though
# they are listed as supported for Clang 3.3 and later.

set(_cmake_oldestSupported "((__clang_major__ * 100) + __clang_minor__) >= 400")

include("${CMAKE_CURRENT_LIST_DIR}/Clang-CXX-TestableFeatures.cmake")

set(AppleClang51_CXX14 "((__clang_major__ * 100) + __clang_minor__) >= 501 && __cplusplus > 201103L")
# http://llvm.org/bugs/show_bug.cgi?id=19242
set(_cmake_feature_test_cxx_attribute_deprecated "${AppleClang51_CXX14}")
# http://llvm.org/bugs/show_bug.cgi?id=19698
set(_cmake_feature_test_cxx_decltype_auto "${AppleClang51_CXX14}")
set(_cmake_feature_test_cxx_digit_separators "${AppleClang51_CXX14}")
# http://llvm.org/bugs/show_bug.cgi?id=19674
set(_cmake_feature_test_cxx_generic_lambdas "${AppleClang51_CXX14}")

set(AppleClang40_CXX11 "${_cmake_oldestSupported} && __cplusplus >= 201103L")
set(_cmake_feature_test_cxx_enum_forward_declarations "${AppleClang40_CXX11}")
set(_cmake_feature_test_cxx_sizeof_member "${AppleClang40_CXX11}")
set(_cmake_feature_test_cxx_extended_friend_declarations "${AppleClang40_CXX11}")
set(_cmake_feature_test_cxx_extern_templates "${AppleClang40_CXX11}")
set(_cmake_feature_test_cxx_func_identifier "${AppleClang40_CXX11}")
set(_cmake_feature_test_cxx_inline_namespaces "${AppleClang40_CXX11}")
set(_cmake_feature_test_cxx_long_long_type "${AppleClang40_CXX11}")
set(_cmake_feature_test_cxx_right_angle_brackets "${AppleClang40_CXX11}")
set(_cmake_feature_test_cxx_variadic_macros "${AppleClang40_CXX11}")

set(AppleClang_CXX98 "${_cmake_oldestSupported} && __cplusplus >= 199711L")
set(_cmake_feature_test_cxx_template_template_parameters "${AppleClang_CXX98}")
