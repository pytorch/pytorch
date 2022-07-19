load("@fbcode_macros//build_defs:python_pytest.bzl", "python_pytest")
load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")
load("@fbsource//tools/build_defs/sandcastle:sandcastle_defs.bzl", "is_sandcastle_machine")

def define_python_unittest(pytest = False, **kwargs):
    build_mode = native.read_config("fbcode", "build_mode_test_label")
    enable_flatbuffer = bool(native.read_config("fbcode", "caffe2_enable_flatbuffer", None))

    PYTORCH_TEST_WITH_ASAN = "1" if ("asan" in build_mode or build_mode == "dev") else "0"

    PYTORCH_TEST_WITH_DEV_DBG_ASAN = "1" if (build_mode == "dev" or "dev-asan" in build_mode or "dbg-asan" in build_mode or "dbgo-asan" in build_mode) else "0"

    PYTORCH_TEST_WITH_TSAN = "1" if ("tsan" in build_mode) else "0"

    PYTORCH_TEST_WITH_UBSAN = "1" if ("ubsan" in build_mode or build_mode == "dev") else "0"

    NO_MULTIPROCESSING_SPAWN = "1" if is_sandcastle_machine() else "0"

    ENABLE_FLATBUFFER = "1" if enable_flatbuffer else "0"

    # indicates we are running in test env.
    # "deepcopy" the 'env: Dict[str, str]'
    kwargs["env"] = dict(kwargs.get("env", {}))
    kwargs["env"]["PYTORCH_TEST"] = "1"
    kwargs["env"]["PYTORCH_TEST_FBCODE"] = "1"
    kwargs["env"]["PYTORCH_TEST_WITH_ASAN"] = PYTORCH_TEST_WITH_ASAN
    kwargs["env"]["PYTORCH_TEST_WITH_DEV_DBG_ASAN"] = PYTORCH_TEST_WITH_DEV_DBG_ASAN
    kwargs["env"]["PYTORCH_TEST_WITH_TSAN"] = PYTORCH_TEST_WITH_TSAN
    kwargs["env"]["PYTORCH_TEST_WITH_UBSAN"] = PYTORCH_TEST_WITH_UBSAN
    kwargs["env"]["NO_MULTIPROCESSING_SPAWN"] = NO_MULTIPROCESSING_SPAWN
    kwargs["env"]["ENABLE_FLATBUFFER"] = ENABLE_FLATBUFFER

    # To speed up TP tests.
    kwargs["env"]["TENSORPIPE_TLS_DATACENTER"] = "test_dc"

    # Run CUDA tests on GPUs
    if kwargs.get("name").endswith("cuda"):
        # "deepcopy" the 'tags: List[str]'
        kwargs["tags"] = list(kwargs.get("tags", []))
        kwargs["tags"].extend([
            "re_opts_capabilities={\"platform\": \"gpu-remote-execution\", \"subplatform\": \"P100\"}",
            "supports_remote_execution",
            "run_as_bundle",
            "tpx:experimental-shard-size-for-bundle=100",
        ])
        kwargs["env"]["PYTORCH_TEST_REMOTE_GPU"] = "1"

    if pytest:
        python_pytest(
            **kwargs
        )
    else:
        python_unittest(
            **kwargs
        )

def define_mp_tests(tests, additional_deps = None, pytest = False, **kwargs):
    # LeakSanitizer doesn't work for python multiprocessing.
    # See https://fb.workplace.com/groups/fbcode/posts/2625521060818050/
    # and https://fb.workplace.com/groups/101100140348621/posts/1278688645923092/
    extra_env = {
        "ASAN_OPTIONS": "detect_leaks=0",
        "CUDA_INJECTION64_PATH": "0",  # resolve kineto TSAN flakiness
    }

    # Serialize test cases since multiple tests running on same GPUs can
    # deadlock or there can be port conflicts.
    if "tags" not in kwargs:
        kwargs["tags"] = []
    if "serialize_test_cases" not in kwargs["tags"]:
        kwargs["tags"].append("serialize_test_cases")
    define_tests(tests, additional_deps, pytest, extra_env, **kwargs)

def define_q_distributed_test(tests, env = None, additional_deps = None, pytest = False, **kwargs):
    define_tests(tests, additional_deps, pytest, env, **kwargs)

def define_tests(tests, additional_deps = None, pytest = False, extra_env = {}, **kwargs):
    if additional_deps == None:
        additional_deps = {}

    provided_tags = kwargs.pop("tags", [])

    env = {
        "DOCS_SRC_DIR": "$(location //caffe2/docs/source:doc_files)",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "SKIP_TEST_BOTTLENECK": "1",
    }
    env.update(extra_env)
    for name, srcs in tests.items():
        tags = list(provided_tags)

        test_deps = ["//caffe2:test-lib"] + additional_deps.get(name, [])
        define_python_unittest(
            pytest,
            name = name,
            srcs = srcs,
            base_module = "",
            compile = "with-source",
            env = env,
            py_version = ">=3.5",
            strip_libpar = True,
            tags = tags,
            deps = test_deps,
            # Depend directly on :libtorch so that tests won't be pruned by the
            # rdep distance heuristic.
            cpp_deps = ["//caffe2:libtorch"],
            runtime_deps = [
                "//caffe2/docs/source:doc_files",
            ],
            **kwargs
        )
