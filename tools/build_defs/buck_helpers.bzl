# Only used for PyTorch open source BUCK build

IGNORED_ATTRIBUTE_PREFIX = [
    "apple",
    "fbobjc",
    "windows",
    "fbandroid",
    "macosx",
]

IGNORED_ATTRIBUTES = [
    "feature",
    "platforms",
    "contacts",
]

# TODO (huydhn): PyTorch OSS is still built with old buck not buck2, and there
# aren't available options https://buck.build/rule/cxx_library.html. This can
# be removed when we migrate OSS to buck2
ONLY_AVAILABLE_IN_BUCK2 = [
    "supports_shlib_interfaces",
]

def filter_attributes(kwgs):
    keys = list(kwgs.keys())

    # drop unncessary attributes
    for key in keys:
        if key in IGNORED_ATTRIBUTES or key in ONLY_AVAILABLE_IN_BUCK2:
            kwgs.pop(key)
        else:
            for invalid_prefix in IGNORED_ATTRIBUTE_PREFIX:
                if key.startswith(invalid_prefix):
                    kwgs.pop(key)
    return kwgs
