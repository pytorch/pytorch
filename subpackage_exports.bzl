load("@fbsource//tools/build_defs:cell_defs.bzl", "get_fbsource_cell")
load("@fbsource//tools/build_defs:fb_native_wrapper.bzl", "fb_native")
load("@fbsource//tools/build_defs:glob_defs.bzl", "subdir_glob")

# Directories under xplat/caffe2 that are their own package, because their BUCK file is shared
# with fbcode through dirsync, but whose files are still built by targets in parent packages.
# A glob stops at a package boundary, so `glob_with_subpackages` / `subdir_glob_with_subpackages`
# add the matching files from this list, as the `export_file` targets that
# `export_files_to_caffe2()` defines in each package.
# oxx targets whose raw headers include these exports set `raw_headers_as_headers_mode = "disabled"`:
# that mode names a raw header by its path within its own package, so a header of
# `torch/headeronly/macros` would become `torch/headeronly/Macros.h` instead of
# `torch/headeronly/macros/Macros.h`.
# Package (relative to xplat/caffe2) -> its files (relative to xplat/caffe2). Keep this in sync
# with the files in those directories.
SUBPACKAGE_FILES = {
    "binaries": [
        "binaries/CMakeLists.txt",
        "binaries/PACKAGE",
        "binaries/aot_model_compiler.cc",
        "binaries/at_launch_benchmark.cc",
        "binaries/benchmark_args.h",
        "binaries/benchmark_helper.cc",
        "binaries/benchmark_helper.h",
        "binaries/caffe2_benchmark.cc",
        "binaries/compare_models_torch.cc",
        "binaries/convert_and_benchmark.cc",
        "binaries/convert_caffe_image_db.cc",
        "binaries/convert_db.cc",
        "binaries/convert_encoded_to_raw_leveldb.cc",
        "binaries/convert_image_to_tensor.cc",
        "binaries/core_overhead_benchmark.cc",
        "binaries/core_overhead_benchmark_gpu.cc",
        "binaries/db_throughput.cc",
        "binaries/dump_operator_names.cc",
        "binaries/fb/defs.bzl",
        "binaries/intra_inter_benchmark.cc",
        "binaries/lite_interpreter_model_load.cc",
        "binaries/load_benchmark_torch.cc",
        "binaries/make_cifar_db.cc",
        "binaries/make_image_db.cc",
        "binaries/make_mnist_db.cc",
        "binaries/optimize_for_mobile.cc",
        "binaries/parallel_info.cc",
        "binaries/predictor_verifier.cc",
        "binaries/print_core_object_sizes_gpu.cc",
        "binaries/print_registered_core_operators.cc",
        "binaries/record_function_benchmark.cc",
        "binaries/run_plan.cc",
        "binaries/run_plan_mpi.cc",
        "binaries/speed_benchmark.cc",
        "binaries/speed_benchmark_torch.cc",
        "binaries/split_db.cc",
        "binaries/tsv_2_proto.cc",
        "binaries/tutorial_blob.cc",
        "binaries/zmq_feeder.cc",
    ],
    "modules": [
        "modules/PACKAGE",
        "modules/module_test/CMakeLists.txt",
        "modules/module_test/module_test_dynamic.cc",
        "modules/observers/CMakeLists.txt",
        "modules/observers/macros.h",
        "modules/observers/net_observer_reporter.h",
        "modules/observers/net_observer_reporter_print.cc",
        "modules/observers/net_observer_reporter_print.h",
        "modules/observers/observer_config.cc",
        "modules/observers/observer_config.h",
        "modules/observers/perf_observer.cc",
        "modules/observers/perf_observer.h",
        "modules/rocksdb/CMakeLists.txt",
        "modules/rocksdb/rocksdb.cc",
    ],
    "torch/headeronly/macros": [
        "torch/headeronly/macros/Export.h",
        "torch/headeronly/macros/Macros.h",
        "torch/headeronly/macros/build.bzl",
        "torch/headeronly/macros/cmake_macros.h.in",
    ],
    "torch/lib/libshm": [
        "torch/lib/libshm/CMakeLists.txt",
        "torch/lib/libshm/PACKAGE",
        "torch/lib/libshm/alloc_info.h",
        "torch/lib/libshm/core.cpp",
        "torch/lib/libshm/err.h",
        "torch/lib/libshm/libshm.h",
        "torch/lib/libshm/manager.cpp",
        "torch/lib/libshm/socket.h",
    ],
}

# Package -> the package that owned its files before it became a package ("" for xplat/caffe2).
# Only globs in that package used to see these files.
SUBPACKAGE_PARENTS = {
    "binaries": "",
    "modules": "",
    "torch/headeronly/macros": "torch/headeronly",
    "torch/lib/libshm": "",
}

def export_files_to_caffe2():
    """Exports every file in this package for the xplat/caffe2 targets that build it. Does nothing in fbcode."""
    if get_fbsource_cell() == "fbcode":
        return
    for f in native.glob(["**/*"], exclude = ["BUCK"]):
        fb_native.export_file(
            name = f,
            mode = "reference",
            visibility = ["//xplat/caffe2/..."],
        )

def _segment_match(pattern, segment):
    parts = pattern.split("*")
    if len(parts) == 1:
        return pattern == segment
    if not segment.startswith(parts[0]) or len(segment) < len(parts[0]) + len(parts[-1]):
        return False
    pos = len(parts[0])
    for part in parts[1:-1]:
        i = segment.find(part, pos)
        if i < 0:
            return False
        pos = i + len(part)
    return segment[pos:].endswith(parts[-1])

def _match(pattern, path):
    ps = pattern.split("/")
    xs = path.split("/")

    # matched[j]: the pattern segments so far match the first j path segments.
    matched = [True] + [False] * len(xs)
    for p in ps:
        if p == "**":
            nxt = list(matched)
            for j in range(1, len(xs) + 1):
                nxt[j] = nxt[j] or nxt[j - 1]
        else:
            nxt = [False] + [matched[j - 1] and _segment_match(p, xs[j - 1]) for j in range(1, len(xs) + 1)]
        matched = nxt
    return matched[len(xs)]

def _literal_prefix(pattern):
    out = []
    for p in pattern.split("/"):
        if "*" in p:
            break
        out.append(p)
    return "/".join(out)

def _subpackage_matches(include, exclude):
    """(path relative to the current package, export label) for subpackage files matching a glob."""
    if get_fbsource_cell() == "fbcode":
        return []
    caller = native.package_name()[len("xplat/caffe2"):].lstrip("/")
    base = caller + "/" if caller else ""
    out = []
    for pkg, files in SUBPACKAGE_FILES.items():
        if SUBPACKAGE_PARENTS[pkg] != caller:
            continue
        for f in files:
            if not f.startswith(base):
                continue
            rel = f[len(base):]
            if any([rel.startswith(_literal_prefix(i)) and _match(i, rel) for i in include]) and not any([_match(e, rel) for e in exclude]):
                out.append((rel, "//xplat/caffe2/{}:{}".format(pkg, f[len(pkg) + 1:])))
    return out

def glob_with_subpackages(include, exclude = [], **kwargs):
    """`glob`, plus the matching files of the packages in SUBPACKAGE_FILES."""
    found = {p: p for p in native.glob(include, exclude = exclude, **kwargs)}
    for rel, label in _subpackage_matches(include, exclude):
        found[rel] = label
    return [found[p] for p in sorted(found.keys())]

def subdir_glob_with_subpackages(glob_specs, exclude = None, prefix = None):
    """`subdir_glob`, plus the matching files of the packages in SUBPACKAGE_FILES."""
    out = subdir_glob(glob_specs, exclude = exclude, prefix = prefix)
    for d, pattern in glob_specs:
        for rel, label in _subpackage_matches([d + "/" + pattern if d else pattern], exclude or []):
            key = rel[len(d) + 1:] if d else rel
            out[prefix + "/" + key if prefix else key] = label
    return out
