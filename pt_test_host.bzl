"""pt_test_host rule definition."""

load("@fbsource//tools/build_defs:fb_xplat_robolectric4_test.bzl", "fb_xplat_robolectric4_test")

def pt_test_host(*args, **kwargs):
    fb_xplat_robolectric4_test(
        *args,
        **kwargs
    )
