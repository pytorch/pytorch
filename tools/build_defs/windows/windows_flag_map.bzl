# Only used for PyTorch open source BUCK build

def windows_convert_gcc_clang_flags(flags = []):
    if read_config("pt", "is_oss", "0") == "0":
        fail("This file is for open source pytorch build. Do not use it in fbsource!")

    # not implemented
    return []
