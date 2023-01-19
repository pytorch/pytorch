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

import unittest
from google.protobuf import text_format
from third_party.com.github.bazelbuild.bazel.src.main.protobuf import crosstool_config_pb2
from tools.migration.ctoolchain_comparator_lib import compare_ctoolchains

from py import mock
try:
  # Python 2
  from cStringIO import StringIO
except ImportError:
  # Python 3
  from io import StringIO


def make_toolchain(toolchain_proto):
  toolchain = crosstool_config_pb2.CToolchain()
  text_format.Merge(toolchain_proto, toolchain)
  return toolchain


class CtoolchainComparatorLibTest(unittest.TestCase):

  def test_string_fields(self):
    first = make_toolchain("""
          toolchain_identifier: "first-id"
          host_system_name: "first-host"
          target_system_name: "first-target"
          target_cpu: "first-cpu"
          target_libc: "first-libc"
          compiler: "first-compiler"
          abi_version: "first-abi"
          abi_libc_version: "first-abi-libc"
          builtin_sysroot: "sysroot"
        """)
    second = make_toolchain("""
          toolchain_identifier: "second-id"
          host_system_name: "second-host"
          target_system_name: "second-target"
          target_cpu: "second-cpu"
          target_libc: "second-libc"
          compiler: "second-compiler"
          abi_version: "second-abi"
          abi_libc_version: "second-abi-libc"
          cc_target_os: "os"
        """)
    error_toolchain_identifier = (
        "Difference in 'toolchain_identifier' field:\n"
        "Value before change:\t'first-id'\n"
        "Value after change:\t'second-id'\n")
    error_host_system_name = ("Difference in 'host_system_name' field:\n"
                              "Value before change:\t'first-host'\n"
                              "Value after change:\t'second-host'\n")
    error_target_system_name = ("Difference in 'target_system_name' field:\n"
                                "Value before change:\t'first-target'\n"
                                "Value after change:\t'second-target'\n")
    error_target_cpu = ("Difference in 'target_cpu' field:\n"
                        "Value before change:\t'first-cpu'\n"
                        "Value after change:\t'second-cpu'\n")
    error_target_libc = ("Difference in 'target_libc' field:\n"
                         "Value before change:\t'first-libc'\n"
                         "Value after change:\t'second-libc'\n")
    error_compiler = ("Difference in 'compiler' field:\n"
                      "Value before change:\t'first-compiler'\n"
                      "Value after change:\t'second-compiler'\n")
    error_abi_version = ("Difference in 'abi_version' field:\n"
                         "Value before change:\t'first-abi'\n"
                         "Value after change:\t'second-abi'\n")
    error_abi_libc_version = ("Difference in 'abi_libc_version' field:\n"
                              "Value before change:\t'first-abi-libc'\n"
                              "Value after change:\t'second-abi-libc'\n")
    error_builtin_sysroot = ("Difference in 'builtin_sysroot' field:\n"
                             "Value before change is set to 'sysroot'\n"
                             "Value after change is not set\n")
    error_cc_target_os = ("Difference in 'cc_target_os' field:\n"
                          "Value before change is not set\n"
                          "Value after change is set to 'os'\n")
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn(error_toolchain_identifier, mock_stdout.getvalue())
      self.assertIn(error_host_system_name, mock_stdout.getvalue())
      self.assertIn(error_target_system_name, mock_stdout.getvalue())
      self.assertIn(error_target_cpu, mock_stdout.getvalue())
      self.assertIn(error_target_libc, mock_stdout.getvalue())
      self.assertIn(error_compiler, mock_stdout.getvalue())
      self.assertIn(error_abi_version, mock_stdout.getvalue())
      self.assertIn(error_abi_libc_version, mock_stdout.getvalue())
      self.assertIn(error_builtin_sysroot, mock_stdout.getvalue())
      self.assertIn(error_cc_target_os, mock_stdout.getvalue())

  def test_tool_path(self):
    first = make_toolchain("""
        tool_path {
          name: "only_first"
          path: "/a/b/c"
        }
        tool_path {
          name: "paths_differ"
          path: "/path/first"
        }
    """)
    second = make_toolchain("""
        tool_path {
          name: "paths_differ"
          path: "/path/second"
        }
        tool_path {
          name: "only_second_1"
          path: "/a/b/c"
        }
        tool_path {
          name: "only_second_2"
          path: "/a/b/c"
        }
    """)
    error_only_first = ("* List before change contains entries for the "
                        "following tools that the list after the change "
                        "doesn't:\n[only_first]\n")
    error_only_second = ("* List after change contains entries for the "
                         "following tools that the list before the change "
                         "doesn't:\n"
                         "[\n"
                         "\tonly_second_1\n"
                         "\tonly_second_2\n"
                         "]\n")
    error_paths_differ = ("* Path for tool 'paths_differ' differs before and "
                          "after the change:\n"
                          "Value before change:\t'/path/first'\n"
                          "Value after change:\t'/path/second'\n")
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn(error_only_first, mock_stdout.getvalue())
      self.assertIn(error_only_second, mock_stdout.getvalue())
      self.assertIn(error_paths_differ, mock_stdout.getvalue())

  def test_make_variable(self):
    first = make_toolchain("""
        make_variable {
          name: "only_first"
          value: "val"
        }
        make_variable {
          name: "value_differs"
          value: "first_value"
        }
    """)
    second = make_toolchain("""
        make_variable {
          name: "value_differs"
          value: "second_value"
        }
        make_variable {
          name: "only_second_1"
          value: "val"
        }
        make_variable {
          name: "only_second_2"
          value: "val"
        }
    """)
    error_only_first = ("* List before change contains entries for the "
                        "following variables that the list after the "
                        "change doesn't:\n[only_first]\n")
    error_only_second = ("* List after change contains entries for the "
                         "following variables that the list before the "
                         "change doesn't:\n"
                         "[\n"
                         "\tonly_second_1\n"
                         "\tonly_second_2\n"
                         "]\n")
    error_value_differs = ("* Value for variable 'value_differs' differs before"
                           " and after the change:\n"
                           "Value before change:\t'first_value'\n"
                           "Value after change:\t'second_value'\n")
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn(error_only_first, mock_stdout.getvalue())
      self.assertIn(error_only_second, mock_stdout.getvalue())
      self.assertIn(error_value_differs, mock_stdout.getvalue())

  def test_cxx_builtin_include_directories(self):
    first = make_toolchain("""
        cxx_builtin_include_directory: "a/b/c"
        cxx_builtin_include_directory: "d/e/f"
    """)
    second = make_toolchain("""
        cxx_builtin_include_directory: "d/e/f"
        cxx_builtin_include_directory: "a/b/c"
    """)
    expect_error = ("Difference in 'cxx_builtin_include_directory' field:\n"
                    "List of elements before change:\n"
                    "[\n"
                    "\ta/b/c\n"
                    "\td/e/f\n"
                    "]\n"
                    "List of elements after change:\n"
                    "[\n"
                    "\td/e/f\n"
                    "\ta/b/c\n"
                    "]\n")
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn(expect_error, mock_stdout.getvalue())

  def test_artifact_name_pattern(self):
    first = make_toolchain("""
        artifact_name_pattern {
          category_name: 'object_file'
          prefix: ''
          extension: '.obj1'
        }
        artifact_name_pattern {
          category_name: 'executable'
          prefix: 'first'
          extension: '.exe'
        }
        artifact_name_pattern {
          category_name: 'dynamic_library'
          prefix: ''
          extension: '.dll'
        }
    """)
    second = make_toolchain("""
        artifact_name_pattern {
          category_name: 'object_file'
          prefix: ''
          extension: '.obj2'
        }
        artifact_name_pattern {
          category_name: 'static_library'
          prefix: ''
          extension: '.lib'
        }
        artifact_name_pattern {
          category_name: 'executable'
          prefix: 'second'
          extension: '.exe'
        }
        artifact_name_pattern {
          category_name: 'interface_library'
          prefix: ''
          extension: '.if.lib'
        }
    """)
    error_only_first = ("* List before change contains entries for the "
                        "following categories that the list after the "
                        "change doesn't:\n[dynamic_library]\n")
    error_only_second = ("* List after change contains entries for the "
                         "following categories that the list before the "
                         "change doesn't:\n"
                         "[\n"
                         "\tinterface_library\n"
                         "\tstatic_library\n"
                         "]\n")
    error_extension_differs = ("* Value for category 'object_file' differs "
                               "before and after the change:\n"
                               "Value before change:"
                               "\tprefix:''"
                               "\textension:'.obj1'\n"
                               "Value after change:"
                               "\tprefix:''"
                               "\textension:'.obj2'\n")
    error_prefix_differs = ("* Value for category 'executable' differs "
                            "before and after the change:\n"
                            "Value before change:"
                            "\tprefix:'first'"
                            "\textension:'.exe'\n"
                            "Value after change:"
                            "\tprefix:'second'"
                            "\textension:'.exe'\n")
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn(error_only_first, mock_stdout.getvalue())
      self.assertIn(error_only_second, mock_stdout.getvalue())
      self.assertIn(error_extension_differs, mock_stdout.getvalue())
      self.assertIn(error_prefix_differs, mock_stdout.getvalue())

  def test_features_not_ordered(self):
    first = make_toolchain("""
        feature {
          name: 'feature1'
        }
        feature {
          name: 'feature2'
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature2'
        }
        feature {
          name: 'feature1'
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("Features not in right order", mock_stdout.getvalue())

  def test_features_missing(self):
    first = make_toolchain("""
        feature {
          name: 'feature1'
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature2'
        }
    """)
    error_only_first = ("* List before change contains entries for the "
                        "following features that the list after the "
                        "change doesn't:\n[feature1]\n")
    error_only_second = ("* List after change contains entries for the "
                         "following features that the list before the "
                         "change doesn't:\n[feature2]\n")
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn(error_only_first, mock_stdout.getvalue())
      self.assertIn(error_only_second, mock_stdout.getvalue())

  def test_feature_enabled(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          enabled: true
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          enabled: false
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after",
                    mock_stdout.getvalue())

  def test_feature_provides(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          provides: 'a'
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          provides: 'b'
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after the change:",
                    mock_stdout.getvalue())

  def test_feature_provides_preserves_order(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          provides: 'a'
          provides: 'b'
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          provides: 'b'
          provides: 'a'
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after the change:",
                    mock_stdout.getvalue())

  def test_feature_implies(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          implies: 'a'
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after the change:",
                    mock_stdout.getvalue())

  def test_feature_implies_preserves_order(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          implies: 'a'
          implies: 'b'
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          implies: 'b'
          implies: 'a'
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after the change:",
                    mock_stdout.getvalue())

  def test_feature_requires_preserves_list_order(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          requires: {
            feature: 'feature1'
          }
          requires: {
            feature: 'feature2'
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          requires: {
            feature: 'feature2'
          }
          requires: {
            feature: 'feature1'
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after the change:",
                    mock_stdout.getvalue())

  def test_feature_requires_ignores_required_features_order(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          requires: {
            feature: 'feature1'
            feature: 'feature2'
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          requires: {
            feature: 'feature2'
            feature: 'feature1'
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("No difference", mock_stdout.getvalue())

  def test_feature_requires_differs(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          requires: {
            feature: 'feature1'
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          requires: {
            feature: 'feature2'
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after the change:",
                    mock_stdout.getvalue())

  def test_action_config_ignores_requires(self):
    first = make_toolchain("""
        action_config {
          config_name: 'config'
          requires: {
            feature: 'feature1'
          }
        }
    """)
    second = make_toolchain("""
        action_config {
          config_name: 'config'
          requires: {
            feature: 'feature2'
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("No difference", mock_stdout.getvalue())

  def test_env_set_actions_differ(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          env_set {
            action: 'a1'
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          env_set: {
            action: 'a1'
            action: 'a2'
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after the change:",
                    mock_stdout.getvalue())

  def test_env_set_ignores_actions_order(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          env_set {
            action: 'a2'
            action: 'a1'
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          env_set: {
            action: 'a1'
            action: 'a2'
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("No difference", mock_stdout.getvalue())

  def test_env_set_env_entries_not_ordered(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          env_set {
            env_entry {
              key: 'k1'
              value: 'v1'
            }
            env_entry {
              key: 'k2'
              value: 'v2'
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          env_set {
            env_entry {
              key: 'k2'
              value: 'v2'
            }
            env_entry {
              key: 'k1'
              value: 'v1'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after the change:",
                    mock_stdout.getvalue())

  def test_env_set_env_entries_differ(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          env_set {
            env_entry {
              key: 'k1'
              value: 'value_first'
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          env_set {
            env_entry {
              key: 'k1'
              value: 'value_second'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after the change:",
                    mock_stdout.getvalue())

  def test_feature_preserves_env_set_order(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          env_set {
            env_entry {
              key: 'first'
              value: 'first'
            }
          }
          env_set {
            env_entry {
              key: 'second'
              value: 'second'
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          env_set {
            env_entry {
              key: 'second'
              value: 'second'
            }
          }
          env_set {
            env_entry {
              key: 'first'
              value: 'first'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after the change:",
                    mock_stdout.getvalue())

  def test_action_config_ignores_env_set(self):
    first = make_toolchain("""
        action_config {
          config_name: 'config'
          env_set {
            env_entry {
              key: 'k1'
              value: 'value_first'
            }
          }
        }
    """)
    second = make_toolchain("""
        action_config {
          config_name: 'config'
          env_set {
            env_entry {
              key: 'k1'
              value: 'value_second'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("No difference", mock_stdout.getvalue())

  def test_env_set_ignores_with_feature_set_order(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          env_set{
            with_feature {
              feature: 'feature1'
            }
            with_feature {
              not_feature: 'feature2'
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          env_set {
            with_feature {
              not_feature: 'feature2'
            }
            with_feature {
              feature: 'feature1'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("No difference", mock_stdout.getvalue())

  def test_env_set_ignores_with_feature_set_lists_order(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          env_set{
            with_feature {
              feature: 'feature1'
              feature: 'feature2'
              not_feature: 'not_feature1'
              not_feature: 'not_feature2'
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          env_set{
            with_feature {
              feature: 'feature2'
              feature: 'feature1'
              not_feature: 'not_feature2'
              not_feature: 'not_feature1'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("No difference", mock_stdout.getvalue())

  def test_flag_set_ignores_actions_order(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          flag_set {
             action: 'a1'
             action: 'a2'
          }
        }
    """)
    second = make_toolchain("""
       feature {
          name: 'feature'
          flag_set {
             action: 'a2'
             action: 'a1'
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("No difference", mock_stdout.getvalue())

  def test_action_config_flag_set_actions_ignored(self):
    first = make_toolchain("""
      action_config {
          config_name: 'config'
          flag_set {
            action: 'a1'
          }
        }
    """)
    second = make_toolchain("""
      action_config {
          config_name: 'config'
          flag_set {
            action: 'a2'
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("No difference", mock_stdout.getvalue())

  def test_flag_set_ignores_with_feature_set_order(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          flag_set {
            with_feature {
              feature: 'feature1'
            }
            with_feature {
              not_feature: 'feature2'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set {
            with_feature {
              feature: 'feature1'
            }
            with_feature {
              not_feature: 'feature2'
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          flag_set {
            with_feature {
              not_feature: 'feature2'
            }
            with_feature {
              feature: 'feature1'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set {
            with_feature {
              not_feature: 'feature2'
            }
            with_feature {
              feature: 'feature1'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("No difference", mock_stdout.getvalue())

  def test_flag_set_ignores_with_feature_set_lists_order(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            with_feature {
              feature: 'feature1'
              feature: 'feature2'
              not_feature: 'not_feature1'
              not_feature: 'not_feature2'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            with_feature {
              feature: 'feature1'
              feature: 'feature2'
              not_feature: 'not_feature1'
              not_feature: 'not_feature2'
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            with_feature {
              feature: 'feature2'
              feature: 'feature1'
              not_feature: 'not_feature2'
              not_feature: 'not_feature1'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            with_feature {
              feature: 'feature2'
              feature: 'feature1'
              not_feature: 'not_feature2'
              not_feature: 'not_feature1'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("No difference", mock_stdout.getvalue())

  def test_flag_set_preserves_flag_group_order(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          flag_set {
            flag_group {
              flag: 'a'
            }
            flag_group {
              flag: 'b'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set {
             flag_group {
               flag: 'a'
             }
             flag_group {
               flag: 'b'
             }
          }
        }
    """)
    second = make_toolchain("""
       feature {
          name: 'feature'
          flag_set {
            flag_group {
              flag: 'b'
            }
            flag_group {
              flag: 'a'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set {
            flag_group {
              flag: 'b'
            }
            flag_group {
              flag: 'a'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after",
                    mock_stdout.getvalue())
      self.assertIn("* Action config 'config' differs before and after",
                    mock_stdout.getvalue())

  def test_flag_group_preserves_flags_order(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              flag: 'flag1'
              flag: 'flag2'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              flag: 'flag1'
              flag: 'flag2'
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              flag: 'flag2'
              flag: 'flag1'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              flag: 'flag2'
              flag: 'flag1'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after",
                    mock_stdout.getvalue())
      self.assertIn("* Action config 'config' differs before and after",
                    mock_stdout.getvalue())

  def test_flag_group_iterate_over_differs(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              iterate_over: 'a'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              iterate_over: 'a'
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              iterate_over: 'b'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              iterate_over: 'b'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after",
                    mock_stdout.getvalue())
      self.assertIn("* Action config 'config' differs before and after",
                    mock_stdout.getvalue())

  def test_flag_group_expand_if_true_differs(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              expand_if_true: 'a'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              expand_if_true: 'a'
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              expand_if_true: 'b'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              expand_if_true: 'b'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after",
                    mock_stdout.getvalue())
      self.assertIn("* Action config 'config' differs before and after",
                    mock_stdout.getvalue())

  def test_flag_group_expand_if_false_differs(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              expand_if_false: 'a'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              expand_if_false: 'a'
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              expand_if_false: 'b'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              expand_if_false: 'b'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after",
                    mock_stdout.getvalue())
      self.assertIn("* Action config 'config' differs before and after",
                    mock_stdout.getvalue())

  def test_flag_group_expand_if_all_available_differs(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              expand_if_all_available: 'a'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              expand_if_all_available: 'a'
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              expand_if_all_available: 'b'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              expand_if_all_available: 'b'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after",
                    mock_stdout.getvalue())
      self.assertIn("* Action config 'config' differs before and after",
                    mock_stdout.getvalue())

  def test_flag_group_expand_if_none_available_differs(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              expand_if_none_available: 'a'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              expand_if_none_available: 'a'
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              expand_if_none_available: 'b'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              expand_if_none_available: 'b'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after",
                    mock_stdout.getvalue())
      self.assertIn("* Action config 'config' differs before and after",
                    mock_stdout.getvalue())

  def test_flag_group_expand_if_all_available_ignores_order(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              expand_if_all_available: 'a'
              expand_if_all_available: 'b'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              expand_if_all_available: 'a'
              expand_if_all_available: 'b'
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              expand_if_all_available: 'b'
              expand_if_all_available: 'a'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              expand_if_all_available: 'b'
              expand_if_all_available: 'a'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("No difference", mock_stdout.getvalue())

  def test_flag_group_expand_if_none_available_ignores_order(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              expand_if_none_available: 'a'
              expand_if_none_available: 'b'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              expand_if_none_available: 'a'
              expand_if_none_available: 'b'
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              expand_if_none_available: 'b'
              expand_if_none_available: 'a'
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              expand_if_none_available: 'b'
              expand_if_none_available: 'a'
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("No difference", mock_stdout.getvalue())

  def test_flag_group_expand_if_equal_differs(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              expand_if_equal {
                variable: 'first'
                value: 'val'
              }
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              expand_if_equal {
                variable: 'first'
                value: 'val'
              }
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              expand_if_equal {
                variable: 'second'
                value: 'val'
              }
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              expand_if_equal {
                variable: 'second'
                value: 'val'
              }
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after",
                    mock_stdout.getvalue())
      self.assertIn("* Action config 'config' differs before and after",
                    mock_stdout.getvalue())

  def test_flag_group_flag_groups_differ(self):
    first = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              flag_group {
                flag: 'a'
                flag: 'b'
              }
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              flag_group {
                flag: 'a'
                flag: 'b'
              }
            }
          }
        }
    """)
    second = make_toolchain("""
        feature {
          name: 'feature'
          flag_set{
            flag_group {
              flag_group {
                flag: 'b'
                flag: 'a'
              }
            }
          }
        }
        action_config {
          config_name: 'config'
          flag_set{
            flag_group {
              flag_group {
                flag: 'b'
                flag: 'a'
              }
            }
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Feature 'feature' differs before and after",
                    mock_stdout.getvalue())
      self.assertIn("* Action config 'config' differs before and after",
                    mock_stdout.getvalue())

  def test_action_configs_not_ordered(self):
    first = make_toolchain("""
        action_config {
          config_name: 'action1'
        }
        action_config {
          config_name: 'action2'
        }
    """)
    second = make_toolchain("""
        action_config {
          config_name: 'action2'
        }
        action_config {
          config_name: 'action1'
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("Action configs not in right order", mock_stdout.getvalue())

  def test_action_configs_missing(self):
    first = make_toolchain("""
        action_config {
          config_name: 'action1'
        }
    """)
    second = make_toolchain("""
        action_config {
          config_name: 'action2'
        }
    """)
    error_only_first = ("* List before change contains entries for the "
                        "following action_configs that the list after the "
                        "change doesn't:\n[action1]\n")
    error_only_second = ("* List after change contains entries for the "
                         "following action_configs that the list before the "
                         "change doesn't:\n[action2]\n")
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn(error_only_first, mock_stdout.getvalue())
      self.assertIn(error_only_second, mock_stdout.getvalue())

  def test_action_config_enabled(self):
    first = make_toolchain("""
        action_config {
          config_name: 'config'
          enabled: true
        }
    """)
    second = make_toolchain("""
        action_config {
          config_name: 'config'
          enabled: false
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Action config 'config' differs before and after",
                    mock_stdout.getvalue())

  def test_action_config_action_name(self):
    first = make_toolchain("""
        action_config {
          config_name: 'config'
          action_name: 'config1'
        }
    """)
    second = make_toolchain("""
        action_config {
          config_name: 'config'
          action_name: 'config2'
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Action config 'config' differs before and after",
                    mock_stdout.getvalue())

  def test_action_config_tool_tool_path_differs(self):
    first = make_toolchain("""
        action_config {
          config_name: 'config'
          tool {
            tool_path: 'path1'
          }
        }
    """)
    second = make_toolchain("""
        action_config {
          config_name: 'config'
          tool {
            tool_path: 'path2'
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Action config 'config' differs before and after",
                    mock_stdout.getvalue())

  def test_action_config_tool_execution_requirements_differ(self):
    first = make_toolchain("""
        action_config {
          config_name: 'config'
          tool {
            execution_requirement: 'a'
          }
        }
    """)
    second = make_toolchain("""
        action_config {
          config_name: 'config'
          tool {
            execution_requirement: 'b'
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Action config 'config' differs before and after",
                    mock_stdout.getvalue())

  def test_action_config_tool_execution_requirements_ignores_order(self):
    first = make_toolchain("""
        action_config {
          config_name: 'config'
          tool {
            execution_requirement: 'a'
            execution_requirement: 'b'
          }
        }
    """)
    second = make_toolchain("""
        action_config {
          config_name: 'config'
          tool {
            execution_requirement: 'b'
            execution_requirement: 'a'
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("No difference", mock_stdout.getvalue())

  def test_action_config_implies_differs(self):
    first = make_toolchain("""
        action_config {
          config_name: 'config'
          implies: 'a'
        }
    """)
    second = make_toolchain("""
        action_config {
          config_name: 'config'
          implies: 'b'
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Action config 'config' differs before and after",
                    mock_stdout.getvalue())

  def test_action_config_implies_preserves_order(self):
    first = make_toolchain("""
        action_config {
          config_name: 'config'
          implies: 'a'
          implies: 'b'
        }
    """)
    second = make_toolchain("""
        action_config {
          config_name: 'config'
          implies: 'b'
          implies: 'a'
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("* Action config 'config' differs before and after",
                    mock_stdout.getvalue())

  def test_unused_tool_path(self):
    first = make_toolchain("""
        tool_path {
          name: "empty"
          path: ""
        }
    """)
    second = make_toolchain("""
        tool_path {
          name: "empty"
          path: "NOT_USED"
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("No difference", mock_stdout.getvalue())

  def test_unused_tool_path_in_tool(self):
    first = make_toolchain("""
        action_config {
          config_name: 'config'
          tool {
            tool_path: ''
          }
        }
    """)
    second = make_toolchain("""
        action_config {
          config_name: 'config'
          tool {
            tool_path: 'NOT_USED'
          }
        }
    """)
    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      compare_ctoolchains(first, second)
      self.assertIn("No difference", mock_stdout.getvalue())

if __name__ == "__main__":
  unittest.main()
