load("@fbsource//tools/build_defs:fb_xplat_cxx_test.bzl", "fb_xplat_cxx_test")
load("@fbsource//tools/build_defs:platform_defs.bzl", "ANDROID", "APPLE", "CXX", "IOS", "MACOSX")
load("@fbsource//xplat/caffe2:c2_defs.bzl", "get_c2_default_cxx_args")

def c2_cxx_test(**kwargs):
    args = get_c2_default_cxx_args()
    args.update(kwargs)
    args["fbandroid_use_instrumentation_test"] = True
    for flag in [
        "macosx_compiler_flags",
        "fbobjc_macosx_configs_override",
        "macosx_frameworks_override",
        "xcode_public_headers_symlinks",
        "macosx_inherited_buck_flags_override",
    ]:
        args.pop(flag, None)
    args["apple_sdks"] = (IOS, MACOSX)
    args["platforms"] = (CXX, APPLE, ANDROID)
    args["contacts"] = ["oncall+ai_infra_mobile_platform@xmail.facebook.com"]
    fb_xplat_cxx_test(**args)
