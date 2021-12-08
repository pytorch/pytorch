load("//third_party:substitution.bzl", "header_template_rule")

rules = struct(
    cc_library=native.cc_library,
    cc_test=native.cc_test,
    filegroup=native.filegroup,
    glob=native.glob,
    header_template=header_template_rule,
    package=native.package,
)
