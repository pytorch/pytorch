import os
import subprocess
import glob

from .env import IS_CONDA, IS_LINUX, CONDA_DIR, check_env_flag, check_negative_env_flag, gather_paths

# On ROCm, RCCL development isn't complete. https://github.com/ROCmSoftwarePlatform/rccl
USE_DISTRIBUTED = not check_negative_env_flag("USE_DISTRIBUTED") and IS_LINUX
USE_GLOO_IBVERBS = False

IB_DEVINFO_CMD = "ibv_devinfo"


def get_command_path(command):
    """
    Helper function that checks if the command exists in the path and gets the
    full path of a given linux command if it exists.
    """
    def executable(command_path):
        return os.path.isfile(command_path) and os.access(command_path, os.X_OK)

    for path in os.environ["PATH"].split(os.pathsep):
        command_path = os.path.join(path, command)
        if executable(command_path):
            return command_path

    return None


def should_build_ib():
    """
    Helper function that detects the system's IB support and returns if we
    should build with IB support.
    """
    ib_util_found = False
    ib_lib_found = False
    ib_header_found = False

    try:
        # If the command doesn't exist, we can directly return instead of
        # making a subprocess call
        full_cmd_path = get_command_path(IB_DEVINFO_CMD)
        if not full_cmd_path:
            ib_util_found = False
        subprocess.check_output([full_cmd_path, "--list"])
        # Here we just would like to simply run the command to test if IB
        # related tools / lib are installed without parsing the output. We
        # will enable IB build as long as the command runs successfully.
        #
        # The output should look like either:
        #
        # > ibv_devinfo --list
        # 0 HCAs founds:
        #
        # or
        #
        # > ibv_devinfo --list
        # 4 HCAs found:
        #   mlx5_3
        #   mlx5_2
        #   mlx5_1
        #   mlx5_0
        ib_util_found = True
    except Exception:
        # We just take all the exceptions here without affecting the build
        ib_util_found = False

    lib_paths = list(filter(bool, [
        "/usr/lib/",
        "/usr/lib/x86_64-linux-gnu/",
        "/usr/lib/powerpc64le-linux-gnu/",
        "/usr/lib/aarch64-linux-gnu/",
    ] + gather_paths([
        "LIBRARY_PATH",
    ]) + gather_paths([
        "LD_LIBRARY_PATH",
    ])))

    include_paths = [
        "/usr/include/",
    ]

    if IS_CONDA:
        lib_paths.append(os.path.join(CONDA_DIR, "lib"))
        include_paths.append(os.path.join(CONDA_DIR, "include"))

    for path in lib_paths:
        if path is None or not os.path.exists(path):
            continue
        ib_libraries = sorted(glob.glob(os.path.join(path, "libibverbs*")))
        if ib_libraries:
            ib_lib_found = True
            break

    for path in include_paths:
        if path is None or not os.path.exists(path):
            continue
        if os.path.exists(os.path.join(path, "infiniband/verbs.h")):
            ib_header_found = True
            break

    return ib_util_found and ib_lib_found and ib_lib_found

if USE_DISTRIBUTED:
    # If the env variable is specified, use the value,
    # otherwise only build with IB when IB support is detected on the system
    if "USE_GLOO_IBVERBS" in os.environ:
        USE_GLOO_IBVERBS = check_env_flag("USE_GLOO_IBVERBS")
    else:
        USE_GLOO_IBVERBS = should_build_ib()
