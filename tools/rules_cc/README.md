# C++ rules for Bazel

* Postsubmit [![Build status](https://badge.buildkite.com/f03592ae2d7d25a2abc2a2ba776e704823fa17fd3e061f5103.svg?branch=master)](https://buildkite.com/bazel/rules-cc)
* Postsubmit + Current Bazel Incompatible flags [![Build status](https://badge.buildkite.com/5ba709cc33e5855078a1f8570adcf8e0a78ea93591bc0b4e81.svg?branch=master)](https://buildkite.com/bazel/rules-cc-plus-bazelisk-migrate)

This repository contains Starlark implementation of C++ rules in Bazel.

The rules are being incrementally converted from their native implementations in the [Bazel source tree](https://source.bazel.build/bazel/+/master:src/main/java/com/google/devtools/build/lib/rules/cpp/).

For the list of C++ rules, see the Bazel
[documentation](https://docs.bazel.build/versions/master/be/overview.html).

# Getting Started

There is no need to use rules from this repository just yet. If you want to use
`rules_cc` anyway, add the following to your `WORKSPACE` file:

```starlark
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_cc",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/TODO"],
    sha256 = "TODO",
)
```

Then, in your `BUILD` files, import and use the rules:

```starlark
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    ...
)
```

# Migration Tools

This repository also contains migration tools that can be used to migrate your
project for Bazel incompatible changes.

## Legacy fields migrator

Script that migrates legacy crosstool fields into features
([incompatible flag](https://github.com/bazelbuild/bazel/issues/6861), 
[tracking issue](https://github.com/bazelbuild/bazel/issues/5883)).

TLDR:

```
bazel run @rules_cc//tools/migration:legacy_fields_migrator -- \
  --input=my_toolchain/CROSSTOOL \
  --inline
```

# Contributing

Bazel and `rules_cc` are the work of many contributors. We appreciate your help!

To contribute, please read the contribution guidelines: [CONTRIBUTING.md](https://github.com/bazelbuild/rules_cc/blob/master/CONTRIBUTING.md).

Note that the `rules_cc` use the GitHub issue tracker for bug reports and feature requests only.
For asking questions see:

* [Stack Overflow](https://stackoverflow.com/questions/tagged/bazel)
* [`rules_cc` mailing list](https://groups.google.com/forum/#!forum/cc-bazel-discuss)
* Slack channel `#cc` on [slack.bazel.build](https://slack.bazel.build)
