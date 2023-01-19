load("@rules_pkg//:pkg.bzl", "pkg_tar")
load("@rules_pkg//releasing:defs.bzl", "print_rel_notes")
load("//cc:version.bzl", "version")

package(
    default_visibility = ["//visibility:private"],
)

# Build the artifact to put on the github release page.
pkg_tar(
    name = "rules_cc-%s" % version,
    srcs = [
        "//:distribution",
        "//cc:distribution",
    ],
    extension = "tar.gz",
    # It is all source code, so make it read-only.
    mode = "0444",
    # Make it owned by root so it does not have the uid of the CI robot.
    owner = "0.0",
    package_dir = ".",
    strip_prefix = ".",
)

print_rel_notes(
    name = "relnotes",
    outs = ["relnotes.txt"],
    repo = "rules_cc",
    version = version,
)
