# Only used for PyTorch open source BUCK build

def is_arvr_mode():
    if read_config("pt", "is_oss", "0") == "0":
        fail("This file is for open source pytorch build. Do not use it in fbsource!")

    return False
