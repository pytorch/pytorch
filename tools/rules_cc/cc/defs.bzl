# Copyright 2018 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Starlark rules for building C++ projects."""

load("//cc/private/rules_impl:cc_flags_supplier.bzl", _cc_flags_supplier = "cc_flags_supplier")
load("//cc/private/rules_impl:compiler_flag.bzl", _compiler_flag = "compiler_flag")

_MIGRATION_TAG = "__CC_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__"

def _add_tags(attrs):
    if "tags" in attrs and attrs["tags"] != None:
        attrs["tags"] = attrs["tags"] + [_MIGRATION_TAG]
    else:
        attrs["tags"] = [_MIGRATION_TAG]
    return attrs

def cc_binary(**attrs):
    """Bazel cc_binary rule.

    https://docs.bazel.build/versions/master/be/c-cpp.html#cc_binary

    Args:
      **attrs: Rule attributes
    """

    # buildifier: disable=native-cc
    native.cc_binary(**_add_tags(attrs))

def cc_test(**attrs):
    """Bazel cc_test rule.

    https://docs.bazel.build/versions/master/be/c-cpp.html#cc_test

    Args:
      **attrs: Rule attributes
    """

    # buildifier: disable=native-cc
    native.cc_test(**_add_tags(attrs))

def cc_library(**attrs):
    """Bazel cc_library rule.

    https://docs.bazel.build/versions/master/be/c-cpp.html#cc_library

    Args:
      **attrs: Rule attributes
    """

    # buildifier: disable=native-cc
    native.cc_library(**_add_tags(attrs))

def cc_import(**attrs):
    """Bazel cc_import rule.

    https://docs.bazel.build/versions/master/be/c-cpp.html#cc_import

    Args:
      **attrs: Rule attributes
    """

    # buildifier: disable=native-cc
    native.cc_import(**_add_tags(attrs))

def cc_proto_library(**attrs):
    """Bazel cc_proto_library rule.

    https://docs.bazel.build/versions/master/be/c-cpp.html#cc_proto_library

    Args:
      **attrs: Rule attributes
    """

    # buildifier: disable=native-cc
    native.cc_proto_library(**_add_tags(attrs))

def fdo_prefetch_hints(**attrs):
    """Bazel fdo_prefetch_hints rule.

    https://docs.bazel.build/versions/master/be/c-cpp.html#fdo_prefetch_hints

    Args:
      **attrs: Rule attributes
    """

    # buildifier: disable=native-cc
    native.fdo_prefetch_hints(**_add_tags(attrs))

def fdo_profile(**attrs):
    """Bazel fdo_profile rule.

    https://docs.bazel.build/versions/master/be/c-cpp.html#fdo_profile

    Args:
      **attrs: Rule attributes
    """

    # buildifier: disable=native-cc
    native.fdo_profile(**_add_tags(attrs))

def cc_toolchain(**attrs):
    """Bazel cc_toolchain rule.

    https://docs.bazel.build/versions/master/be/c-cpp.html#cc_toolchain

    Args:
      **attrs: Rule attributes
    """

    # buildifier: disable=native-cc
    native.cc_toolchain(**_add_tags(attrs))

def cc_toolchain_suite(**attrs):
    """Bazel cc_toolchain_suite rule.

    https://docs.bazel.build/versions/master/be/c-cpp.html#cc_toolchain_suite

    Args:
      **attrs: Rule attributes
    """

    # buildifier: disable=native-cc
    native.cc_toolchain_suite(**_add_tags(attrs))

def objc_library(**attrs):
    """Bazel objc_library rule.

    https://docs.bazel.build/versions/master/be/objective-c.html#objc_library

    Args:
      **attrs: Rule attributes
    """

    # buildifier: disable=native-cc
    native.objc_library(**_add_tags(attrs))

def objc_import(**attrs):
    """Bazel objc_import rule.

    https://docs.bazel.build/versions/master/be/objective-c.html#objc_import

    Args:
      **attrs: Rule attributes
    """

    # buildifier: disable=native-cc
    native.objc_import(**_add_tags(attrs))

def cc_flags_supplier(**attrs):
    """Bazel cc_flags_supplier rule.

    Args:
      **attrs: Rule attributes
    """
    _cc_flags_supplier(**_add_tags(attrs))

def compiler_flag(**attrs):
    """Bazel compiler_flag rule.

    Args:
      **attrs: Rule attributes
    """
    _compiler_flag(**_add_tags(attrs))
