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
]

def filter_attributes(kwgs):
    keys = list(kwgs.keys())

    # drop unncessary attributes
    for key in keys:
        if key in IGNORED_ATTRIBUTES:
            kwgs.pop(key)
        else:
            for invalid_prefix in IGNORED_ATTRIBUTE_PREFIX:
                if key.startswith(invalid_prefix):
                    kwgs.pop(key)
    return kwgs

# maps known fbsource deps to OSS deps
DEPS_MAP = {
    "//third-party/FP16:FP16": "//third_party:FP16",
    "//third-party/FXdiv:FXdiv": "//third_party:FXdiv",
    "//third-party/XNNPACK:XNNPACK": "//third_party:XNNPACK",
    "//third-party/clog:clog": "//third_party:clog",
    "//third-party/cpuinfo:cpuinfo": "//third_party:cpuinfo",
    "//third-party/fmt:fmt": "//third_party:fmt",
    "//third-party/glog:glog": "//third_party:glog",
    "//third-party/psimd:psimd": "//third_party:psimd",
    "//third-party/pthreadpool:pthreadpool": "//third_party:pthreadpool",
    "//third-party/pthreadpool:pthreadpool_header": "//third_party:pthreadpool_header",
    "//third-party/ruy:ruy_xplat_lib": "//third_party:ruy_lib",
}

# map fbsource deps to OSS deps
def to_oss_deps(deps = []):
    new_deps = []
    for dep in deps:
        new_deps += map_deps(dep)
    return new_deps

def map_deps(dep):
    # remove @fbsource prefix
    if dep.startswith("@fbsource"):
        dep = dep[len("@fbsource"):]

    # ignore all fbsource linker_lib targets
    if dep.startswith("//xplat/third-party/linker_lib"):
        return []

    # map targets in caffe2 root folder. Just use relative path
    if dep.startswith("//xplat/caffe2:"):
        return [dep[len("//xplat/caffe2"):]]

    # map targets in caffe2 subfolders
    if dep.startswith("//xplat/caffe2/"):
        return ["//" + dep[len("//xplat/caffe2/"):]]

    # map other known targets
    if dep in DEPS_MAP:
        return DEPS_MAP[dep]

    # drop other unknown deps
    return []
