load(":toolchain.bzl", "cxx_toolchain", "python_toolchain", "python_bootstrap_toolchain")

cxx_toolchain(
    name="cxx",
    visibility=["PUBLIC"],
)

python_toolchain(
    name="python",
    visibility=["PUBLIC"],
)

# the bootstrap toolchain is used
# to poentially initialize other
# toolchains
python_bootstrap_toolchain(
    name="python_bootstrap",
    visibility=["PUBLIC"],
)
