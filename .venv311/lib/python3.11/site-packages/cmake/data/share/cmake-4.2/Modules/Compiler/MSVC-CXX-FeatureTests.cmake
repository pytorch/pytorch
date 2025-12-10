# Reference: https://docs.microsoft.com/en-us/cpp/visual-cpp-language-conformance
# https://blogs.msdn.microsoft.com/vcblog/2015/06/19/c111417-features-in-vs-2015-rtm/
# https://blogs.msdn.microsoft.com/vcblog/2013/12/02/c1114-core-language-features-in-vs-2013-and-the-nov-2013-ctp/
# https://blogs.msdn.microsoft.com/vcblog/2011/09/12/c11-features-in-visual-c-11/

set(_cmake_oldestSupported "_MSC_VER >= 1600")

# VS 2017 v15.3 fixes support for incomplete decltypes
# https://docs.microsoft.com/en-us/cpp/cpp-conformance-improvements-2017#update_153
set(_cmake_feature_test_cxx_decltype_incomplete_return_types "_MSC_VER >= 1911")

# VS 2017 v15.3 fixes support for "N3652 Extended constexpr"
set(_cmake_feature_test_cxx_relaxed_constexpr "_MSC_VER >= 1911")

# VS 2017 Preview introduces support for aggregate initializers.
set(_cmake_feature_test_cxx_aggregate_default_initializers "_MSC_FULL_VER >= 190024406")

# VS 2015 Update 2 introduces support for variable templates.
# https://www.visualstudio.com/en-us/news/vs2015-update2-vs.aspx
set(_cmake_feature_test_cxx_variable_templates "_MSC_FULL_VER >= 190023918")

set(MSVC_2015 "_MSC_VER >= 1900")
set(_cmake_feature_test_cxx_alignas "${MSVC_2015}")
set(_cmake_feature_test_cxx_alignof "${MSVC_2015}")
set(_cmake_feature_test_cxx_attributes "${MSVC_2015}")
set(_cmake_feature_test_cxx_attribute_deprecated "${MSVC_2015}")
set(_cmake_feature_test_cxx_binary_literals "${MSVC_2015}")
set(_cmake_feature_test_cxx_constexpr "${MSVC_2015}")
set(_cmake_feature_test_cxx_decltype_auto "${MSVC_2015}")
set(_cmake_feature_test_cxx_digit_separators "${MSVC_2015}")
set(_cmake_feature_test_cxx_func_identifier "${MSVC_2015}")
set(_cmake_feature_test_cxx_nonstatic_member_init "${MSVC_2015}")
# Microsoft calls this 'rvalue references v3'
set(_cmake_feature_test_cxx_defaulted_move_initializers "${MSVC_2015}")
set(_cmake_feature_test_cxx_generic_lambdas "${MSVC_2015}")
set(_cmake_feature_test_cxx_inheriting_constructors "${MSVC_2015}")
set(_cmake_feature_test_cxx_inline_namespaces "${MSVC_2015}")
set(_cmake_feature_test_cxx_lambda_init_captures "${MSVC_2015}")
set(_cmake_feature_test_cxx_noexcept "${MSVC_2015}")
set(_cmake_feature_test_cxx_return_type_deduction "${MSVC_2015}")
set(_cmake_feature_test_cxx_sizeof_member "${MSVC_2015}")
set(_cmake_feature_test_cxx_thread_local "${MSVC_2015}")
set(_cmake_feature_test_cxx_unicode_literals "${MSVC_2015}")
set(_cmake_feature_test_cxx_unrestricted_unions "${MSVC_2015}")
set(_cmake_feature_test_cxx_user_literals "${MSVC_2015}")
set(_cmake_feature_test_cxx_reference_qualified_functions "${MSVC_2015}")
# "The copies and moves don't interact precisely like the Standard says they
# should. For example, deletion of moves is specified to also suppress
# copies, but Visual C++ in Visual Studio 2013 does not."
# http://blogs.msdn.com/b/vcblog/archive/2014/11/17/c-11-14-17-features-in-vs-2015-preview.aspx
# lists this as 'partial' in 2013
set(_cmake_feature_test_cxx_deleted_functions "${MSVC_2015}")

# http://blogs.msdn.com/b/vcblog/archive/2014/11/17/c-11-14-17-features-in-vs-2015-preview.aspx
# Note 1. While previous version of VisualStudio said they supported these
# they silently produced bad code, and are now marked as having partial
# support in previous versions. The footnote says the support will be complete
# in MSVC 2015, so support the feature for that version, assuming that is true.
# The blog post also says that VS 2013 Update 3 generates an error in cases
# that previously produced bad code.
set(_cmake_feature_test_cxx_generalized_initializers "_MSC_FULL_VER >= 180030723")

set(MSVC_2013 "_MSC_VER >= 1800")
set(_cmake_feature_test_cxx_alias_templates "${MSVC_2013}")
# Microsoft now states they support contextual conversions in 2013 and above.
# See footnote 6 at:
# http://blogs.msdn.com/b/vcblog/archive/2014/11/17/c-11-14-17-features-in-vs-2015-preview.aspx
set(_cmake_feature_test_cxx_contextual_conversions "${MSVC_2013}")
set(_cmake_feature_test_cxx_default_function_template_args "${MSVC_2013}")
set(_cmake_feature_test_cxx_defaulted_functions "${MSVC_2013}")
set(_cmake_feature_test_cxx_delegating_constructors "${MSVC_2013}")
set(_cmake_feature_test_cxx_explicit_conversions "${MSVC_2013}")
set(_cmake_feature_test_cxx_raw_string_literals "${MSVC_2013}")
set(_cmake_feature_test_cxx_uniform_initialization "${MSVC_2013}")
# Support is documented, but possibly partly broken:
# https://msdn.microsoft.com/en-us/library/hh567368.aspx
# http://thread.gmane.org/gmane.comp.lib.boost.devel/244986/focus=245333
set(_cmake_feature_test_cxx_variadic_templates "${MSVC_2013}")

set(MSVC_2012 "_MSC_VER >= 1700")
set(_cmake_feature_test_cxx_enum_forward_declarations "${MSVC_2012}")
set(_cmake_feature_test_cxx_final "${MSVC_2012}")
set(_cmake_feature_test_cxx_range_for "${MSVC_2012}")
set(_cmake_feature_test_cxx_strong_enums "${MSVC_2012}")

set(MSVC_2010 "_MSC_VER >= 1600")
set(_cmake_feature_test_cxx_auto_type "${MSVC_2010}")
set(_cmake_feature_test_cxx_decltype "${MSVC_2010}")
set(_cmake_feature_test_cxx_extended_friend_declarations "${MSVC_2010}")
set(_cmake_feature_test_cxx_extern_templates "${MSVC_2010}")
set(_cmake_feature_test_cxx_lambdas "${MSVC_2010}")
set(_cmake_feature_test_cxx_local_type_template_args "${MSVC_2010}")
set(_cmake_feature_test_cxx_long_long_type "${MSVC_2010}")
set(_cmake_feature_test_cxx_nullptr "${MSVC_2010}")
set(_cmake_feature_test_cxx_override "${MSVC_2010}")
set(_cmake_feature_test_cxx_right_angle_brackets "${MSVC_2010}")
set(_cmake_feature_test_cxx_rvalue_references "${MSVC_2010}")
set(_cmake_feature_test_cxx_static_assert "${MSVC_2010}")
set(_cmake_feature_test_cxx_template_template_parameters "${MSVC_2010}")
set(_cmake_feature_test_cxx_trailing_return_types "${MSVC_2010}")
set(_cmake_feature_test_cxx_variadic_macros "${MSVC_2010}")

# Unset all the variables that we don't need exposed.
# _cmake_oldestSupported is required by WriteCompilerDetectionHeader
set(MSVC_2017)
set(MSVC_2015)
set(MSVC_2013)
set(MSVC_2012)
set(MSVC_2010)
