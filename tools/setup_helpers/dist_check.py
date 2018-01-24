import os
import platform
import subprocess

from .env import check_env_flag

IS_WINDOWS = (platform.system() == 'Windows')

WITH_DISTRIBUTED = not check_env_flag('NO_DISTRIBUTED') and not IS_WINDOWS
WITH_DISTRIBUTED_MW = WITH_DISTRIBUTED and check_env_flag('WITH_DISTRIBUTED_MW')
WITH_GLOO_IBVERBS = False

IB_DEVINFO_CMD = "ibv_devinfo"


def get_command_path(command):
    """
    Helper function that checks if the command exists in the path and gets the
    full path of a given linux command if it exists.
    """
    def excutable(command_path):
        return os.path.isfile(command_path) and os.access(command_path, os.X_OK)

    for path in os.environ["PATH"].split(os.pathsep):
        command_path = os.path.join(path, command)
        if excutable(command_path):
            return command_path

    return None


def should_build_ib():
    """
    Helper function that detects the system's IB support and returns if we
    should build with IB support.
    """
    try:
        # If the command doesn't exist, we can directly return instead of
        # making a subprocess call
        full_cmd_path = get_command_path(IB_DEVINFO_CMD)
        if not full_cmd_path:
            return False
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
        return True
    except Exception:
        # We just take all the exceptions here without affecting the build
        return False


WITH_GLOO_IBVERBS = WITH_DISTRIBUTED and (should_build_ib() or
                                          check_env_flag("WITH_GLOO_IBVERBS"))
