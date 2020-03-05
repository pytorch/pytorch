def fb_internal_c10_args():
    return {}

def define_fb_internal_targets():
    pass


def exported_preprocessor_flags():
    _EXPORTED_PREPROCESSOR_FLAGS = [
        # We do not use cmake, hence we will use the following macro to bypass
        # the cmake_macros.h file.
        "-DC10_USING_CUSTOM_GENERATED_MACROS",
        # We will use minimal glog to reduce binary size.
        "-DC10_USE_MINIMAL_GLOG",
        # No NUMA on mobile.  Disable it so it doesn't mess with host tests.
        "-DC10_DISABLE_NUMA"
    ]

    return _EXPORTED_PREPROCESSOR_FLAGS


def c10_default_exported_preprocessor_flags():
    return exported_preprocessor_flags() + [
        # C10 will extensively use exceptions so we will enable it.
        "-fexceptions",
        # for static string objects in c10.
        "-Wno-global-constructors",
    ]


def fb_internal_test_c10_args():
    return {}
