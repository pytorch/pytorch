load("@fbsource//tools/build_defs:buckconfig.bzl", "read_bool")

is_sgx = read_bool("fbcode", "sgx_mode", False)
