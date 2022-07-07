load("@fbsource//tools/build_defs:testpilot_defs.bzl", "special_tags")
load(
    "//caffe2/test:defs.bzl",
    "define_python_unittest",
)

# These distributed tests need custom environment variables
def define_distributed_test(**kwargs):
    # LeakSanitizer doesn't work for python multiprocessing.
    # See https://fb.workplace.com/groups/fbcode/posts/2625521060818050/
    # and https://fb.workplace.com/groups/101100140348621/posts/1278688645923092/
    kwargs["env"]["ASAN_OPTIONS"] = "detect_leaks=0"

    # Resolve kineto TSAN flakiness
    kwargs["env"]["CUDA_INJECTION64_PATH"] = "0"
    define_python_unittest(
        base_module = "",
        main_module = "fb.test_distributed_trap",
        py_version = ">=3.5",
        tags = [special_tags.run_as_bundle],
        deps = [
            "//caffe2:test-lib",
            "//caffe2:torch",
            "//caffe2/torch/fb/rendezvous:zeus",
            "//pytorch/vision:torchvision",
        ],
        external_deps = [
            ("numpy", None),
            ("scipy", None),
        ],
        **kwargs
    )

def define_c10d_distributed_test(srcs, **kwargs):
    srcs.extend(["fb/test_distributed_trap.py"])
    define_distributed_test(
        srcs = srcs + native.glob(["data/*.py"]),
        **kwargs
    )
