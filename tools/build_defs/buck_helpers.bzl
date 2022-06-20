# Only used for PyTorch open source BUCK build
load("//tools/build_defs:type_defs.bzl", "is_dict", "is_list")

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

def filter_files(files):
    print(files)
    if is_list(files):
        new_files = []
        for file in files:
            if file.startswith("fb/") or file.startswith("caffe2/fb/") or file.startswith("torch/fb/"):
                continue
            else:
                new_files.append(file)
    else:
        new_files = {}
        for key, file in files.items():
            if file.startswith("fb/") or file.startswith("caffe2/fb/") or file.startswith("torch/fb/"):
                continue
            else:
                new_files[key] = file
    return new_files

# map fbsource deps to OSS deps
def to_oss_deps(deps = []):
    new_deps = []
    for dep in deps:
        new_deps += map_deps(dep)
    return new_deps

def map_deps(dep):
    # keep relative root targets
    if dep.startswith(":"):
        return [dep]

    # remove @fbsource prefix
    if dep.startswith("@fbsource"):
        dep = dep[len("@fbsource"):]

    # remove xplat/caffe2 prefix
    if dep.startswith("//xplat/caffe2:"):
        dep = dep[len("//xplat/caffe2"):]

    # remove xplat/caffe2/ prefix
    if dep.startswith("//xplat/caffe2/"):
        dep = dep[len("//xplat/caffe2/"):]

    # ignore all fbsource linker_lib targets
    if dep.startswith("//xplat/third-party/linker_lib:"):
        return []

    # ignore all folly libraries
    if dep.startswith("//xplat/folly:"):
        return []

    # return unknown deps for easy debugging
    return [dep]
