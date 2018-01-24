import os
import subprocess
import re


WITH_IB_DEVICES = False
IB_DEVINFO_CMD = "ibv_devinfo"


def get_command_path(command):
    """
    Helper function that get the full path of a given linux command
    """
    def excutable(command_path):
        return os.path.isfile(command_path) and os.access(command_path, os.X_OK)

    for path in os.environ["PATH"].split(os.pathsep):
        command_path = os.path.join(path, command)
        if excutable(command_path):
            return command_path

    return None


def detect_ib_devices():
    """
    Helper function that detects if there are Infiniband devices on the host,
    and returns the number of IB devices detected or None for failure to detect
    """
    try:
        full_cmd_path = get_command_path(IB_DEVINFO_CMD)
        if not full_cmd_path:
            return None
        out = subprocess.check_output([full_cmd_path, "--list"])
        # find the first line of the output
        # The outpyt should be either:
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
        first_line = out.decode().split('\n')[0]
        res = re.findall("\d+", first_line)
        if len(res) != 1:
            raise Exception("-- IB_detect: unexpected parsing error while "
                            "trying to find the number of available devices.")
        return int(res[0])

    except Exception as ex:
        # We just take all the exceptions here without affecting the build
        print("-- IB_detect: encountered an exception: {}".format(str(ex)))
        return None


num_ib_devices = detect_ib_devices()

if num_ib_devices is None:
    print("-- IB_detect: unable to detect IB devices, "
          "compiling with no IB support by default unless overridden "
          "by WITH_GLOO_IBVERBS")

elif num_ib_devices > 0:
    print("-- IB_detect: {} IB devices detected, compiling with IB support."
          .format(num_ib_devices))
    WITH_IB_DEVICES = True

else:
    print("-- IB_detect: no IB device detected, compiling with no IB support "
          "by default unless overridden by WITH_GLOO_IBVERBS")
