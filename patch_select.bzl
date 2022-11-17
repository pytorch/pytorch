# this file is currently needed due to an issue with buck2
# that causes the include directive to not be applied to
# bzl files
def select(conditions):
    if read_config("pt", "is_oss", "0") == "0":
        fail("This file is for open source pytorch build. Do not use it in fbsource!")

    return conditions["DEFAULT"]