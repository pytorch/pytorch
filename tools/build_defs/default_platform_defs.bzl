# Only used for PyTorch open source BUCK build
# @lint-ignore-every BUCKRESTRICTEDSYNTAX

def compose_platform_setting_list(settings):
    """Settings object:
    os/cpu pair: should be valid key, or at most one part can be wildcard.
    flags: the values added to the compiler flags
    """
    if read_config("pt", "is_oss", "0") == "0":
        fail("This file is for open source pytorch build. Do not use it in fbsource!")

    result = []
    for setting in settings:
        result.append([
            "^{}-{}$".format(setting["os"], setting["cpu"]),
            setting["flags"],
        ])
    return result
