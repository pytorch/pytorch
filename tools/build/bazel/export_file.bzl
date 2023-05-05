load("//lib:visibility.bzl", "from_buck_visibility")

def export_file(file, visibility):
    native.exports_files([file], visibility = from_buck_visibility(visibility))
