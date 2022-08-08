load("@fbsource//tools/build_defs:expect.bzl", "expect")
load(
    "@fbsource//tools/build_defs/apple:build_mode_defs.bzl",
    "is_production_build",
)

###############################################################################
# Check if we need to strip glog.
def _get_strip_glog_config():
    c2_strip_glog = native.read_config("caffe2", "strip_glog", "1")
    expect(
        c2_strip_glog in ("0", "1"),
        c2_strip_glog,
    )
    return bool(int(c2_strip_glog))

# For iOS production builds (and all Android builds), strip GLOG logging to
# save size. We can disable by setting caffe2.strip_glog=0 in .buckconfig.local.
def get_fbobjc_strip_glog_flags():
    if is_production_build() or _get_strip_glog_config():
        return ["-UGOOGLE_STRIP_LOG", "-DGOOGLE_STRIP_LOG=3"]
    else:
        return ["-UGOOGLE_STRIP_LOG"]

def get_fbandroid_strip_glog_flags():
    if _get_strip_glog_config():
        return ["-UGOOGLE_STRIP_LOG", "-DGOOGLE_STRIP_LOG=1"]
    else:
        return []
