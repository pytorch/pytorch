# @lint-ignore-every FBCODEBZLADDLOADS
load("//tools/build_defs:glob_defs.bzl", "subdir_glob")

# shared by internal and OSS BUCK
def define_tools_targets(
        python_binary,
        python_library,
        python_test,
        third_party,
        torchgen_deps,
        contacts = []):
    python_library(
        name = "substitutelib",
        srcs = ["substitute.py"],
        base_module = "",
    )

    python_binary(
        name = "substitute",
        main_module = "substitute",
        visibility = ["PUBLIC"],
        deps = [
            ":substitutelib",
        ],
    )

    python_library(
        name = "jit",
        srcs = glob([
            "jit/*.py",
            "jit/templates/*",
        ]),
        base_module = "tools",
        visibility = ["PUBLIC"],
        deps = [
            torchgen_deps,
        ],
    )

    python_binary(
        name = "gen_unboxing_bin",
        main_module = "tools.jit.gen_unboxing",
        visibility = [
            "PUBLIC",
        ],
        deps = [
            ":jit",
        ],
    )

    python_library(
        name = "gen_selected_mobile_ops_header",
        srcs = ["lite_interpreter/gen_selected_mobile_ops_header.py"],
        base_module = "tools",
        visibility = ["PUBLIC"],
    )

    python_library(
        name = "gen_oplist_lib",
        srcs = subdir_glob([
            ("code_analyzer", "gen_oplist.py"),
            ("code_analyzer", "gen_op_registration_allowlist.py"),
        ]),
        base_module = "tools.code_analyzer",
        tests = [
            ":gen_oplist_test",
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":gen_selected_mobile_ops_header",
            torchgen_deps,
            third_party("pyyaml"),
        ],
    )

    python_binary(
        name = "gen_oplist",
        main_module = "tools.code_analyzer.gen_oplist",
        visibility = ["PUBLIC"],
        deps = [
            ":gen_oplist_lib",
        ],
    )

    python_library(
        name = "gen_operators_yaml_lib",
        srcs = subdir_glob([
            ("code_analyzer", "gen_operators_yaml.py"),
            ("code_analyzer", "gen_op_registration_allowlist.py"),
        ]),
        base_module = "",
        tests = [
            ":gen_operators_yaml_test",
        ],
        deps = [
            third_party("pyyaml"),
            torchgen_deps,
        ],
    )

    python_binary(
        name = "gen_operators_yaml",
        main_module = "gen_operators_yaml",
        visibility = ["PUBLIC"],
        deps = [
            ":gen_operators_yaml_lib",
        ],
    )

    python_library(
        name = "autograd",
        srcs = glob(["autograd/*.py"]),
        base_module = "tools",
        resources = [
            "autograd/deprecated.yaml",
            "autograd/derivatives.yaml",
            "autograd/templates/ADInplaceOrViewType.cpp",
            "autograd/templates/Functions.cpp",
            "autograd/templates/Functions.h",
            "autograd/templates/TraceType.cpp",
            "autograd/templates/VariableType.cpp",
            "autograd/templates/VariableType.h",
            "autograd/templates/ViewFuncs.cpp",
            "autograd/templates/ViewFuncs.h",
            "autograd/templates/annotated_fn_args.py.in",
            "autograd/templates/python_enum_tag.cpp",
            "autograd/templates/python_fft_functions.cpp",
            "autograd/templates/python_functions.cpp",
            "autograd/templates/python_functions.h",
            "autograd/templates/python_linalg_functions.cpp",
            "autograd/templates/python_nested_functions.cpp",
            "autograd/templates/python_nn_functions.cpp",
            "autograd/templates/python_return_types.h",
            "autograd/templates/python_return_types.cpp",
            "autograd/templates/python_sparse_functions.cpp",
            "autograd/templates/python_special_functions.cpp",
            "autograd/templates/python_torch_functions.cpp",
            "autograd/templates/python_variable_methods.cpp",
            "autograd/templates/variable_factories.h",
        ],
        visibility = ["PUBLIC"],
        deps = [
            third_party("pyyaml"),
            torchgen_deps,
        ],
    )

    python_library(
        name = "generate_code",
        srcs = [
            "setup_helpers/generate_code.py",
        ],
        base_module = "tools",
        deps = [
            ":autograd",
            ":jit",
            torchgen_deps,
        ],
    )

    python_binary(
        name = "generate_code_bin",
        main_module = "tools.setup_helpers.generate_code",
        # Windows does not support inplace:
        # https://github.com/facebook/buck/issues/2161.
        #
        # Note that //arvr/mode/embedded/win/clang-aarch64-release sets
        # its target platform to
        # ovr_config//platform/embedded:clang-aarch64-linux-release, hence
        # that is why we are selecting that OS to trigger this behavior.
        package_style = select({
            "DEFAULT": "inplace",
            "ovr_config//os:linux-arm64": "standalone",
        }),
        visibility = ["PUBLIC"],
        # Because Windows does not support inplace packaging, we need to
        # ensure it is unzipped before executing it, otherwise it will not
        # be able to find any resources using path manipulation.
        #
        # See note above about why the OS is Linux here and not Windows.
        zip_safe = select({
            "DEFAULT": True,
            "ovr_config//os:linux-arm64": False,
        }),
        deps = [
            ":generate_code",
        ],
    )

    python_library(
        name = "gen-version-header-lib",
        srcs = [
            "setup_helpers/gen_version_header.py",
        ],
        base_module = "",
        deps = [],
    )

    python_binary(
        name = "gen-version-header",
        main_module = "setup_helpers.gen_version_header",
        visibility = ["PUBLIC"],
        deps = [
            ":gen-version-header-lib",
        ],
    )

    python_library(
        name = "gen_aten_vulkan_spv_lib",
        srcs = [
            "gen_vulkan_spv.py",
        ],
        base_module = "tools",
        deps = [
            torchgen_deps,
        ],
    )

    python_binary(
        name = "gen_aten_vulkan_spv_bin",
        main_module = "tools.gen_vulkan_spv",
        visibility = [
            "PUBLIC",
        ],
        deps = [
            ":gen_aten_vulkan_spv_lib",
        ],
    )

    python_test(
        name = "vulkan_codegen_test",
        srcs = [
            "test/test_vulkan_codegen.py",
        ],
        contacts = contacts,
        visibility = ["PUBLIC"],
        deps = [
            ":gen_aten_vulkan_spv_lib",
        ],
    )

    python_test(
        name = "selective_build_test",
        srcs = [
            "test/test_selective_build.py",
        ],
        contacts = contacts,
        visibility = ["PUBLIC"],
        deps = [
            torchgen_deps,
        ],
    )

    python_test(
        name = "gen_oplist_test",
        srcs = [
            "test/gen_oplist_test.py",
        ],
        contacts = contacts,
        visibility = ["PUBLIC"],
        deps = [
            ":gen_oplist_lib",
        ],
    )

    python_test(
        name = "gen_operators_yaml_test",
        srcs = [
            "test/gen_operators_yaml_test.py",
        ],
        visibility = ["PUBLIC"],
        contacts = contacts,
        deps = [
            ":gen_operators_yaml_lib",
        ],
    )

    python_test(
        name = "test_codegen",
        srcs = [
            "test/test_codegen.py",
        ],
        contacts = contacts,
        visibility = ["PUBLIC"],
        deps = [
            torchgen_deps,
            ":autograd",
        ],
    )

    python_test(
        name = "test_torchgen_executorch",
        srcs = [
            "test/test_executorch_gen.py",
            "test/test_executorch_signatures.py",
            "test/test_executorch_types.py",
            "test/test_executorch_unboxing.py",
        ],
        contacts = contacts,
        visibility = ["PUBLIC"],
        deps = [
            torchgen_deps,
        ],
    )
